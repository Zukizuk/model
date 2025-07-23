import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

def load_engine(engine_path):
    """Load TensorRT engine"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    return engine, context

def allocate_buffers(engine):
    """Allocate GPU memory"""
    inputs = []
    outputs = []
    bindings = []
    
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(engine.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocess frame for YOLO"""
    # Resize
    resized = cv2.resize(frame, input_size)
    # Convert BGR to RGB and normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    # CHW format and add batch dimension
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)

def simple_nms(boxes, scores, score_threshold=0.5, nms_threshold=0.4):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    # Filter by score
    valid_indices = scores > score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    
    if len(boxes) == 0:
        return []
    
    # Convert to x1,y1,x2,y2 format for NMS
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    boxes_xyxy = np.column_stack([x1, y1, x2, y2])
    
    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), 
        score_threshold, nms_threshold
    )
    
    if len(indices) > 0:
        return indices.flatten()
    return []

def run_inference_on_video(engine_path, video_path, output_path=None):
    """Run YOLO inference on video"""
    
    # Load engine
    engine, context = load_engine(engine_path)
    inputs, outputs, bindings = allocate_buffers(engine)
    stream = cuda.Stream()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video if specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess
        input_data = preprocess_frame(frame)
        
        # Copy to GPU and run inference
        start_time = time.time()
        
        np.copyto(inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Get output shape and reshape
        output_shape = context.get_binding_shape(1)
        output = outputs[0]['host'].reshape(output_shape)
        
        # Simple post-processing (adjust based on your YOLO version)
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # Extract detections (assuming standard YOLO format)
        if output.shape[1] >= 5:  # Has at least x,y,w,h,conf
            boxes = output[:, :4]  # x,y,w,h
            scores = output[:, 4]  # confidence
            
            # Scale boxes to original frame size
            scale_x = width / 640
            scale_y = height / 640
            
            boxes[:, 0] *= scale_x  # x
            boxes[:, 1] *= scale_y  # y
            boxes[:, 2] *= scale_x  # w
            boxes[:, 3] *= scale_y  # h
            
            # Simple NMS
            keep_indices = simple_nms(boxes, scores)
            
            # Draw detections
            for idx in keep_indices:
                x, y, w, h = boxes[idx]
                score = scores[idx]
                
                # Convert to corner coordinates
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{score:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show progress
        if frame_count % 30 == 0:
            avg_fps = frame_count / total_time
            print(f"Frame {frame_count}/{total_frames}, Avg FPS: {avg_fps:.1f}")
        
        # Save frame if output specified
        if out:
            out.write(frame)
        
        # Display frame (optional - comment out for headless)
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time
    print(f"Processing complete! Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    engine_path = "weights.engine"
    video_path = "video.mp4"
    output_path = "output_video.mp4"
    
    run_inference_on_video(engine_path, video_path, output_path)