# Real Robot Migration Spec: Place Onto Target Detected Pose

## Goal
On the real robot, replicate the simulation workflow: **pick a specific part (`source_id`) and place it onto the detected pose of a target workpiece (`target_id`)**.

This document defines the **contracts**, **math**, and **configuration schema** so implementation can be done with minimal ambiguity.

---

## 1) Contracts (Perception → Manipulation)

### 1.1 Pose topic contract
Perception publishes object poses to:
- `/Current_OBJ_position_<id>` (`geometry_msgs/PoseStamped`)

**Semantic meaning**
- Message pose is the object pose **in the robot base frame**: `^bT_obj`

**Required fields**
- `header.frame_id` MUST be the base frame, typically `base_link` (or configured `--base_frame`)
- Quaternion format is ROS standard **xyzw**
- Timestamp MUST be valid; the controller will reject stale target poses

### 1.2 Frame contract
Choose one base frame for the whole system:
- `base_frame`: default `base_link`

Perception must output `^baseT_obj` directly (preferred), i.e. do TF first (or use a static extrinsic fallback) and publish in `base_frame`.

### 1.3 “Local offset” contract (key decision)
For placement we define:
- `place_offset_local[source_id]` is a 3D vector `[x, y, z]` in **target workpiece frame** (target local frame).

Interpretation:
- final place position is “target detected position + (target rotation * offset_local)”

---

## 2) Placement math (authoritative)

Given:
- Target pose `^bT_t` from `/Current_OBJ_position_<target_id>`
- `offset_local = place_offset_local[source_id]` (expressed in target frame `t`)

Compute:
- `R = R(^bT_t)` (3x3), `p = p(^bT_t)` (3x1)
- `p_place = p + R * offset_local`
- `q_place = q(^bT_t)` (**fully follow target orientation**)

Output placement pose (base frame):
- `place_pose.position = p_place`
- `place_pose.orientation = q_place`

Approach pose:
- `approach_pose = place_pose` but `approach_pose.position.z += place_approach_offset`

Notes:
- Keep a `place_z_tweak` scalar for hardware settling / gripper geometry compensation:
  - `p_place.z += place_z_tweak`

---

## 3) Motion sequence (Pick → PlaceOnTarget)

### 3.1 Pick sequence (high-level)
1. Get `/Current_OBJ_position_<source_id>` (freshness gate)
2. Compute grasp approach / pre-grasp pose (implementation-specific)
3. Move to pre-grasp (MoveIt first, SDK fallback optional)
4. Descend to grasp
5. Close gripper
6. Lift by `lift_height`

### 3.2 Place-on-target sequence (high-level)
1. Get `/Current_OBJ_position_<target_id>` (freshness gate)
2. Compute `place_pose` using Section 2 math
3. Execute:
   - move(approach_pose)
   - move(place_pose)
   - open_gripper
   - move(approach_pose)  (retreat)

### 3.3 Freshness + stability gating (real robot safety)
Minimum recommended checks before planning:
- `now - msg.header.stamp <= pose_max_age_sec` (e.g. 0.3–1.0s depending on FPS)
- Optional: reject large jumps vs last accepted pose:
  - position jump > `pos_jump_m` (e.g. 0.05m)
  - orientation jump > `rot_jump_deg` (e.g. 20°)

---

## 4) YAML configuration schema (recommended)

Create one YAML file as the single source of truth (paths are examples):
- `config/pick_place.yaml`

Schema:
```yaml
frames:
  base_frame: base_link
  camera_frame: d435_depth_optical_frame  # optional, for TF lookup

topics:
  pose_topic_prefix: /Current_OBJ_position_

motion:
  pose_max_age_sec: 0.5
  approach_distance: 0.10
  lift_height: 0.10
  place_approach_offset: 0.08
  place_z_tweak: 0.01

objects:
  "1":
    place_offset_local: [0.035839, 0.111555, 0.1240]   # in TARGET frame
    grasp_library: demo_data/ship_data/object1_grasp.yml
  "2":
    place_offset_local: [0.085839, 0.111555, 0.1240]
    grasp_library: demo_data/ship_data/object2_grasp.yml
```

Rules:
- Keys can be strings in YAML; controller should parse to `int`.
- `place_offset_local` is mandatory for every `source_id` you plan to place.

---

## 5) Recommended refactor split (to avoid sim/real duplication)

### 5.1 Sequencer (environment-agnostic)
Single place to encode the workflow:
- `pick(source_id)`
- `place_on_target(source_id, target_id)`
- `run_sequence(source_ids, target_id)` (CLI “last id is target” like simulation)

Sequencer depends on small interfaces only:
- `PoseSource.get_pose(id) -> PoseStamped`
- `MotionBackend.move_pose(pose: PoseStamped) -> bool`
- `GripperBackend.open()/close() -> bool`

### 5.2 Backends (environment-specific)
Implementations can differ but must match the same interface:
- Simulation:
  - gripper via action (`GripperCommand`)
  - attach/detach via LinkAttacher services (optional)
- Real robot:
  - gripper via SDK (`RM_controller.set_gripper`)
  - attach/detach should be **MoveIt planning scene attach** or no-op
  - MoveIt start-state sync from hardware joint readout

---

## 6) Perception node alignment (must-have for this workflow)

To support “place onto target detected pose”, perception must be consistent:
- Prefer TF lookup (`base_frame <- camera_frame`) and publish object poses in base frame directly.
- If TF is unavailable, fallback to a static extrinsic YAML (avoid hardcoding inside `cam_2_base_transform.py`).

Implementation reference:
- Simulation workspace perception already has a robust pattern:
  - configurable camera topics, depth decoding to meters, TF lookup with fallback
  - publishes `/Current_OBJ_position_<id>` with `header.frame_id = output_frame`

Real workspace perception should match that behavior and remove remaining hard-coded topics.

---

## 7) Acceptance checklist (practical)

### 7.1 Pose sanity
- `ros2 topic echo /Current_OBJ_position_<id>` shows `frame_id: base_link`
- Position is stable in meters; orientation is finite and normalized

### 7.2 Place math sanity
With robot disabled, log computed placement pose:
- Verify: if `offset_local = [0,0,0]` then `place_pose == target_pose`
- Rotate target by 90° about Z: `R*offset_local` should rotate accordingly

### 7.3 Sequence test
Manual CLI test mode:
- Input: `1 5` means “pick 1, place onto target 5”
- Input: `1 2 5` means “pick 1 then 2, each placed onto target 5”

---

## 8) Implementation to-do list (for the coding AI)

1. Perception: ensure `/Current_OBJ_position_<id>` is base-frame `PoseStamped`
2. Config: load YAML `objects[source_id].place_offset_local`
3. Controller: add `move_to_target(source_id, target_id)` using Section 2 math
4. Sequencer: unify manual CLI (last id is target) for the real robot controller
5. Safety: add pose freshness gating + basic jump rejection

