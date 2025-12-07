#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { callVroom, Depot, Vehicle, Stop, VroomVrpOutput } from "./vroomClient.js";

const server = new Server(
  {
    name: "vroom-vrp-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "vroom_vrp",
        description: "Solve Vehicle Routing Problem using VROOM solver",
        inputSchema: {
          type: "object",
          properties: {
            depot: {
              type: "object",
              properties: {
                id: { type: "string" },
                lat: { type: "number" },
                lon: { type: "number" },
              },
              required: ["id", "lat", "lon"],
            },
            vehicles: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  id: { type: "string" },
                  depot_id: { type: "string" },
                  capacity: { type: "number" },
                },
                required: ["id", "depot_id"],
              },
            },
            stops: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  id: { type: "string" },
                  lat: { type: "number" },
                  lon: { type: "number" },
                  demand: { type: "number" },
                  service_time_s: { type: "number" },
                },
                required: ["id", "lat", "lon"],
              },
            },
          },
          required: ["depot", "vehicles", "stops"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "vroom_vrp") {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  try {
    const args = request.params.arguments as {
      depot: Depot;
      vehicles: Vehicle[];
      stops: Stop[];
    };

    // Validate input
    if (!args.depot || !args.vehicles || !args.stops) {
      throw new Error("Missing required fields: depot, vehicles, or stops");
    }

    if (!Array.isArray(args.vehicles) || args.vehicles.length === 0) {
      throw new Error("vehicles must be a non-empty array");
    }

    if (!Array.isArray(args.stops)) {
      throw new Error("stops must be an array");
    }

    // Call VROOM
    const result: VroomVrpOutput = await callVroom(args);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : String(error);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            { error: errorMessage },
            null,
            2
          ),
        },
      ],
      isError: true,
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("VROOM VRP MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});

