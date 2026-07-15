"use client";

import { useEffect, useRef, useState } from "react";

const MAPS_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

declare global {
  interface Window {
    __posyMapsLoader?: Promise<void>;
  }
}

function loadGoogleMaps(key: string): Promise<void> {
  if (typeof google !== "undefined" && google.maps?.places) {
    return Promise.resolve();
  }
  if (!window.__posyMapsLoader) {
    window.__posyMapsLoader = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&libraries=places&loading=async`;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("maps failed to load"));
      document.head.appendChild(script);
    });
  }
  return window.__posyMapsLoader;
}

type Defaults = {
  line1?: string | null;
  line2?: string | null;
  city?: string | null;
  state?: string | null;
  postal_code?: string | null;
};

// Address inputs for the saveAddress form. When a Google Maps key is
// configured, line1 becomes a Places autocomplete: picking a suggestion
// fills street/city/state/zip. Apt/unit is never touched by autocomplete
// (suggestions don't carry unit numbers) and every field stays hand-
// editable after a pick.
export function AddressFields({ defaults }: { defaults: Defaults }) {
  const line1Ref = useRef<HTMLInputElement>(null);
  const [city, setCity] = useState(defaults.city ?? "");
  const [state, setState] = useState(defaults.state ?? "");
  const [zip, setZip] = useState(defaults.postal_code ?? "");

  useEffect(() => {
    if (!MAPS_KEY || !line1Ref.current) return;
    let autocomplete: google.maps.places.Autocomplete | undefined;

    loadGoogleMaps(MAPS_KEY)
      .then(() => {
        if (!line1Ref.current) return;
        autocomplete = new google.maps.places.Autocomplete(line1Ref.current, {
          types: ["address"],
          componentRestrictions: { country: "us" },
          fields: ["address_components"],
        });
        autocomplete.addListener("place_changed", () => {
          const components = autocomplete?.getPlace()?.address_components;
          if (!components) return;
          const part = (type: string, short = false) => {
            const c = components.find((component) =>
              component.types.includes(type),
            );
            return c ? (short ? c.short_name : c.long_name) : "";
          };
          const street = `${part("street_number")} ${part("route")}`.trim();
          if (line1Ref.current && street) line1Ref.current.value = street;
          setCity(part("locality") || part("sublocality") || part("postal_town"));
          setState(part("administrative_area_level_1", true));
          setZip(part("postal_code"));
        });
      })
      .catch(() => {
        // Autocomplete is an enhancement; the plain form keeps working.
      });

    return () => {
      if (autocomplete) google.maps.event.clearInstanceListeners(autocomplete);
    };
  }, []);

  return (
    <>
      <label className="stack">
        <span>Street address</span>
        <input
          ref={line1Ref}
          className="input"
          name="line1"
          required
          placeholder={
            MAPS_KEY ? "Start typing an address…" : "123 Oak Street"
          }
          defaultValue={defaults.line1 ?? ""}
          autoComplete="off"
        />
      </label>
      <label className="stack">
        <span>Apt / unit (optional)</span>
        <input
          className="input"
          name="line2"
          defaultValue={defaults.line2 ?? ""}
        />
      </label>
      <label className="stack">
        <span>City</span>
        <input
          className="input"
          name="city"
          required
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />
      </label>
      <div className="stack row-on-wide">
        <label className="stack" style={{ flex: 1 }}>
          <span>State</span>
          <input
            className="input"
            name="state"
            value={state}
            onChange={(e) => setState(e.target.value)}
          />
        </label>
        <label className="stack" style={{ flex: 1 }}>
          <span>ZIP</span>
          <input
            className="input"
            name="postal_code"
            value={zip}
            onChange={(e) => setZip(e.target.value)}
          />
        </label>
      </div>
    </>
  );
}
