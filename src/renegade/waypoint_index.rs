use std::collections::HashMap;
use bit_vec::BitVec;

struct WaypointIndex<I> {
    waypoints : Vec<(I, I)>,
}

impl<I> WaypointIndex<I> {
    fn new(data : Vec<I>, dist : &Fn(I, I) -> f64, waypoint_count : u8) -> WaypointIndex<I> {
        todo!();
    }

    fn bit_vec(&self, item : I) -> BitVec {
        todo!();
    }
}