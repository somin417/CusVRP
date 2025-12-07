export interface Depot {
  id: string;
  lat: number;
  lon: number;
}

export interface Vehicle {
  id: string;
  depot_id: string;
  capacity?: number;
}

export interface Stop {
  id: string;
  lat: number;
  lon: number;
  demand?: number;
  service_time_s?: number;
}

export interface RouteStep {
  stop_id: string | null;
  lat: number;
  lon: number;
  arrival_s: number;
  distance_m: number;
}

export interface Route {
  vehicle_id: string;
  total_duration_s: number;
  total_distance_m: number;
  steps: RouteStep[];
}

export interface VroomVrpOutput {
  routes: Route[];
  summary: {
    total_duration_s: number;
    total_distance_m: number;
    unassigned_stops: string[];
  };
}

interface VroomVehicle {
  id: number;
  start: [number, number];
  end: [number, number];
  capacity?: number[];
}

interface VroomJob {
  id: number;
  location: [number, number];
  service?: number;
  amount?: number[];
}

interface VroomRequest {
  vehicles: VroomVehicle[];
  jobs: VroomJob[];
}

interface VroomStep {
  type: number; // 0=start, 1=job, 2=end
  location?: [number, number];
  arrival?: number;
  distance?: number;
  id?: number;
}

interface VroomRoute {
  vehicle: number;
  steps: VroomStep[];
  cost: number;
  service: number;
  duration: number;
  distance: number;
}

interface VroomResponse {
  code: number;
  summary: {
    cost: number;
    service: number;
    duration: number;
    distance: number;
    unassigned?: number;
  };
  unassigned?: Array<{ id: number; reason?: number }>;
  routes: VroomRoute[];
}

export async function callVroom(input: {
  depot: Depot;
  vehicles: Vehicle[];
  stops: Stop[];
}): Promise<VroomVrpOutput> {
  const baseUrl = process.env.VROOM_BASE_URL || "http://localhost:3000/";
  const url = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  // Map to VROOM format
  const vroomVehicles: VroomVehicle[] = input.vehicles.map((v, idx) => ({
    id: idx + 1,
    start: [input.depot.lon, input.depot.lat], // VROOM uses [lon, lat]
    end: [input.depot.lon, input.depot.lat],
    capacity: v.capacity ? [v.capacity] : undefined,
  }));

  const vroomJobs: VroomJob[] = input.stops.map((s, idx) => ({
    id: idx + 1,
    location: [s.lon, s.lat], // VROOM uses [lon, lat]
    service: s.service_time_s || 0,
    amount: s.demand ? [s.demand] : undefined,
  }));

  const request: VroomRequest = {
    vehicles: vroomVehicles,
    jobs: vroomJobs,
  };

  // Call VROOM API
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`VROOM API error: ${response.status} ${errorText}`);
  }

  const vroomResponse = await response.json() as VroomResponse;

  if (vroomResponse.code !== 0) {
    throw new Error(`VROOM solver error: code ${vroomResponse.code}`);
  }

  // Map VROOM response to our output format
  const stopMap = new Map(input.stops.map((s, idx) => [idx + 1, s]));
  const vehicleMap = new Map(input.vehicles.map((v, idx) => [idx + 1, v]));

  const routes: Route[] = vroomResponse.routes.map((r) => {
    const vehicle = vehicleMap.get(r.vehicle);
    if (!vehicle) throw new Error(`Vehicle ${r.vehicle} not found`);

    const steps: RouteStep[] = r.steps.map((step) => {
      if (step.type === 0 || step.type === 2) {
        // Start or end at depot
        return {
          stop_id: null,
          lat: input.depot.lat,
          lon: input.depot.lon,
          arrival_s: step.arrival || 0,
          distance_m: step.distance || 0,
        };
      } else if (step.type === 1 && step.id) {
        // Job/stop
        const stop = stopMap.get(step.id);
        if (!stop) throw new Error(`Stop ${step.id} not found`);
        return {
          stop_id: stop.id,
          lat: stop.lat,
          lon: stop.lon,
          arrival_s: step.arrival || 0,
          distance_m: step.distance || 0,
        };
      }
      throw new Error(`Invalid step type: ${step.type}`);
    });

    return {
      vehicle_id: vehicle.id,
      total_duration_s: r.duration,
      total_distance_m: r.distance,
      steps,
    };
  });

  const unassignedStops: string[] = (vroomResponse.unassigned || []).map((u) => {
    const stop = stopMap.get(u.id);
    return stop?.id || `unknown_${u.id}`;
  });

  return {
    routes,
    summary: {
      total_duration_s: vroomResponse.summary.duration,
      total_distance_m: vroomResponse.summary.distance,
      unassigned_stops: unassignedStops,
    },
  };
}

