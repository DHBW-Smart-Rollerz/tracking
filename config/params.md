object_tracking_node:
  ros__parameters:
    # --- Node settings ---
    debug: false
    state: 1
    export_timing_csv: false
    crossing_tracking_enabled: false

    # --- Subscriber topics ---
    image_subscriber: "/camera/image/undistorted"
    object_detection_subscriber: "/object_detection/object"
    sign_detection_subscriber: "/object_detection/sign"
    crossing_detection_subscriber: "/crossing_detection/result"
    state_subscriber: "/tracking/state"

    # --- Publisher topics ---
    state_publisher: "/tracking/state"

    # --- Common parameters ---
    dt: 0.1
    publish_interval_ms: 15      
    tracker_step_interval_ms: 40 
    max_x: 3000.0
    max_y: 2000.0 
    
    # --- Object tracker (moving objects: cars, pedestrians) ---
    object_max_age: 32 
    object_min_hits: 6             
    object_min_age: 5             
    object_max_distance: 4.5     
    object_q_pos: 50.0      
    object_q_vel: 180.0       
    object_r_pos: 100.0  
    object_r_dist_ref: 1400.0
    object_sigma_pos_init: 100.0    
    object_sigma_vel_init: 500.0   
    object_duplicate_distance: 0.0

    # --- Sign tracker (traffic signs) ---
    sign_max_age: 35    
    sign_min_hits: 3   
    sign_min_age: 2    
    sign_max_distance: 7 
    sign_q_pos: 50.0 
    sign_q_vel: 250.0 
    sign_r_pos: 150.0            
    sign_r_dist_ref: 1400.0 
    sign_sigma_pos_init: 100.0  
    sign_sigma_vel_init: 500.0
    sign_duplicate_distance: 0.0  

    # --- Crossing tracker (lane lines at intersections) ---
    crossing_max_age: 3  
    crossing_min_hits: 1  
    crossing_min_age: 0 
    crossing_max_distance: 4.5
    crossing_q_pos: 20.0
    crossing_q_vel: 30.0
    crossing_r_pos: 80.0
    crossing_r_dist_ref: 1100.0
    crossing_sigma_pos_init: 300.0
    crossing_sigma_vel_init: 300.0
    crossing_duplicate_distance: 0.0
