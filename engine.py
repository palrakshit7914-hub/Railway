import json
import os
from datetime import datetime

# --- 1. DATA LOADING FUNCTIONS ---

def load_json_file(file_path):
    """Safely loads a JSON file with error handling."""
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON format in {file_path}: {e}")
        return []

def load_all_department_requests():
    """Combines maintenance requests from TMS, SMMS, and TDMS JSON files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    tms_data = load_json_file(os.path.join(data_dir, 'tms_data.json'))
    smms_data = load_json_file(os.path.join(data_dir, 'smms_data.json'))
    tdms_data = load_json_file(os.path.join(data_dir, 'tdms_data.json'))

    return tms_data + smms_data + tdms_data

def load_coa_train_data():
    """Loads COA running train timetable."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_json_file(os.path.join(base_dir, 'data', 'coa_data.json'))


# --- 2. OVERLAP & COLLISION LOGIC ---

def is_spatial_overlap(req1, req2):
    """Returns True if two requests share the same track section and kilometer range."""
    if req1.get('section_id') != req2.get('section_id'):
        return False
    # Check if physical track KM ranges intersect
    return max(req1['start_km'], req2['start_km']) < min(req1['end_km'], req2['end_km'])

def check_train_conflicts(joint_block, train_list):
    """Checks if the planned block conflicts with any high-priority running trains."""
    conflicts = []
    for train in train_list:
        if train.get('section_id') == joint_block['section_id']:
            # Example check: Flag if high-priority trains (e.g. Rajdhani) operate on the section
            if train.get('priority') == 1:
                conflicts.append({
                    "train_number": train.get('train_number'),
                    "train_name": train.get('train_name'),
                    "warning": "High priority train on line during planned window!"
                })
    return conflicts


# --- 3. CORE AI OPTIMIZATION ENGINE ---

def generate_optimized_schedule():
    """Processes raw departmental requests and returns merged joint blocks + efficiency metrics."""
    raw_requests = load_all_department_requests()
    train_data = load_coa_train_data()

    if not raw_requests:
        return {
            "status": "error",
            "message": "No maintenance data found. Please check your JSON files inside data/ folder."
        }

    optimized_blocks = []
    processed_ids = set()

    for i, req1 in enumerate(raw_requests):
        req1_id = req1.get('request_id')
        if req1_id in processed_ids:
            continue

        # Create base Joint Block structure
        joint_block = {
            "joint_block_id": f"BLK-OPT-{len(optimized_blocks) + 1:03d}",
            "section_id": req1.get('section_id'),
            "departments_involved": [req1.get('department')],
            "combined_request_ids": [req1_id],
            "work_summary": [f"[{req1.get('department')}] {req1.get('work_type')}"],
            "start_km": req1.get('start_km'),
            "end_km": req1.get('end_km'),
            "required_duration_mins": req1.get('requested_duration_mins', 60),
            "urgency": req1.get('urgency', 'MEDIUM'),
            "scheduled_start_time": req1.get('preferred_start_time')
        }
        processed_ids.add(req1_id)

        # Look for overlapping requests from other departments
        for j, req2 in enumerate(raw_requests):
            req2_id = req2.get('request_id')
            if i != j and req2_id not in processed_ids:
                if is_spatial_overlap(req1, req2):
                    # Merge req2 into the Joint Block
                    joint_block["departments_involved"].append(req2.get('department'))
                    joint_block["combined_request_ids"].append(req2_id)
                    joint_block["work_summary"].append(f"[{req2.get('department')}] {req2.get('work_type')}")
                    
                    # Expand KM range to cover both works
                    joint_block["start_km"] = min(joint_block["start_km"], req2.get('start_km'))
                    joint_block["end_km"] = max(joint_block["end_km"], req2.get('end_km'))
                    
                    # Duration is the max duration required + a 15 min buffer for team coordination
                    joint_block["required_duration_mins"] = max(
                        joint_block["required_duration_mins"], 
                        req2.get('requested_duration_mins', 60)
                    ) + 15
                    
                    # If any task is HIGH urgency, upgrade the whole block
                    if req2.get('urgency') == 'HIGH':
                        joint_block["urgency"] = 'HIGH'

                    processed_ids.add(req2_id)

        # Check train conflicts for this block
        joint_block["potential_train_conflicts"] = check_train_conflicts(joint_block, train_data)
        optimized_blocks.append(joint_block)

    # Calculate Savings Metrics for Dashboard / Pitch Deck
    total_unoptimized_mins = sum(r.get('requested_duration_mins', 0) for r in raw_requests)
    total_optimized_mins = sum(b['required_duration_mins'] for b in optimized_blocks)
    time_saved_mins = max(0, total_unoptimized_mins - total_optimized_mins)

    return {
        "status": "success",
        "total_raw_requests": len(raw_requests),
        "total_joint_blocks_created": len(optimized_blocks),
        "metrics": {
            "unoptimized_track_closure_hours": round(total_unoptimized_mins / 60, 2),
            "optimized_track_closure_hours": round(total_optimized_mins / 60, 2),
            "hours_saved_for_trains": round(time_saved_mins / 60, 2),
            "efficiency_gain_percent": round((time_saved_mins / total_unoptimized_mins) * 100, 1) if total_unoptimized_mins > 0 else 0.0
        },
        "optimized_blocks": optimized_blocks
    }


# --- 4. TERMINAL EXECUTION TEST ---

if __name__ == "__main__":
    print("\n--- Running AI Block Optimization Engine Test ---\n")
    output = generate_optimized_schedule()
    
    if output.get("status") == "success":
        print(f" Success! Processed {output['total_raw_requests']} requests into {output['total_joint_blocks_created']} Joint Block(s).")
        print(f" Hours Saved for Train Operations: {output['metrics']['hours_saved_for_trains']} Hours ({output['metrics']['efficiency_gain_percent']}% Capacity Gain)")
        print("\nGenerated Joint Blocks Preview:")
        print(json.dumps(output["optimized_blocks"], indent=2))
    else:
        print(f" Error: {output.get('message')}")