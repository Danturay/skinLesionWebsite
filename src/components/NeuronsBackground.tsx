// src/components/NeuronsBackground.tsx
"use client";

import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import { useCallback } from "react";

export const NeuronsBackground = () => {
  const particlesInit = useCallback(async (engine: any) => {
    await loadSlim(engine);
  }, []);

  return (
    <Particles
      id="neuron-background"
      init={particlesInit}
      options={{
        fullScreen: { enable: false },
        background: { color: "transparent" },
        particles: {
  number: {
    value: 100,
    density: {
      enable: true,
      area: 750,
    },
  },
  color: {
    value: ["#FFD700", "#FFAC33", "#FFC300"], // Gold shades
  },
  links: {
    enable: true,
    distance: 100,
    color: "#FFD700", // Golden link
    opacity: 0.4,
    width: 1.5,
    triangles: {
      enable: true,
      opacity: 0.05,
      color: "#FFD700",
    },
  },
  move: {
    enable: true,
    speed: 0.2,
    direction: "none",
    random: true,
    straight: false,
    outModes: {
      default: "bounce",
    },
  },
  size: {
    value: { min: 1, max: 3 },
  },
  opacity: {
    value: 0.8,
    animation: {
      enable: true,
      speed: 0.1,
      minimumValue: 0.3,
      sync: false,
    },
  },
  shape: {
    type: "circle",
  },
},

        interactivity: {
          events: {
            onHover: {
              enable: true,
              mode: "grab",
            },
            resize: true,
          },
          modes: {
            grab: {
              distance: 150,
              line_linked: {
                opacity: 0.5,
              },
            },
          },
        },
        detectRetina: true,
      }}
      className="absolute inset-0 -z-10"
    />
  );
};
