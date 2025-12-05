"""Check default vehicle speeds in highway-env"""
import highway_env
import gymnasium

# Create environment
env = gymnasium.make('highway-v0')
config = env.unwrapped.config

print("=" * 50)
print("HIGHWAY-ENV DEFAULT SPEED CONFIGURATION")
print("=" * 50)

print("\n[Config Parameters]")
print(f"  initial_lane_id: {config.get('initial_lane_id')}")
print(f"  ego_spacing: {config.get('ego_spacing')}")
print(f"  reward_speed_range: {config.get('reward_speed_range')} m/s")
if config.get('reward_speed_range'):
    low, high = config.get('reward_speed_range')
    print(f"                     = [{low*3.6:.0f}, {high*3.6:.0f}] km/h")

# Reset to get vehicles
obs, info = env.reset()
road = env.unwrapped.road
ego = env.unwrapped.vehicle

print("\n[Ego Vehicle]")
print(f"  Class: {ego.__class__.__name__}")
print(f"  Initial speed: {ego.speed:.1f} m/s = {ego.speed*3.6:.0f} km/h")
print(f"  MAX_SPEED: {ego.MAX_SPEED:.1f} m/s = {ego.MAX_SPEED*3.6:.0f} km/h")
print(f"  MIN_SPEED: {getattr(ego, 'MIN_SPEED', 0):.1f} m/s")

# Other vehicles
print("\n[Other Vehicles (IDM Behavior)]")
other_vehicles = [v for v in road.vehicles if v != ego]
if other_vehicles:
    speeds = [v.speed for v in other_vehicles]
    print(f"  Count: {len(other_vehicles)}")
    print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} m/s")
    print(f"              = {min(speeds)*3.6:.0f} - {max(speeds)*3.6:.0f} km/h")
    print(f"  Average speed: {sum(speeds)/len(speeds):.1f} m/s = {sum(speeds)/len(speeds)*3.6:.0f} km/h")
    
    # Check IDM params
    sample_v = other_vehicles[0]
    print(f"\n  Vehicle class: {sample_v.__class__.__name__}")
    if hasattr(sample_v, 'target_speed'):
        print(f"  target_speed: {sample_v.target_speed:.1f} m/s = {sample_v.target_speed*3.6:.0f} km/h")
    
    print("\n  Individual vehicle speeds:")
    for i, v in enumerate(other_vehicles[:10]):
        target = getattr(v, 'target_speed', None)
        target_str = f" (target: {target:.1f} m/s)" if target else ""
        print(f"    V{i+1}: {v.speed:.1f} m/s = {v.speed*3.6:.0f} km/h{target_str}")

# Check simulation params
print("\n[Simulation Parameters]")
print(f"  simulation_frequency: {config.get('simulation_frequency')} Hz")
print(f"  policy_frequency: {config.get('policy_frequency')} Hz")
print(f"  duration: {config.get('duration')} s")

# Available speed-related config options
print("\n[Speed-Related Config Keys]")
for key in sorted(config.keys()):
    if 'speed' in key.lower() or 'velocity' in key.lower():
        print(f"  {key}: {config.get(key)}")

env.close()
print("\n" + "=" * 50)
