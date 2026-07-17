"use client";

import { useEffect, useRef, useState } from "react";
import { Flex, Text, TextField } from "@radix-ui/themes";

const MAPS_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

declare global {
  interface Window {
    __posyMapsLoader?: Promise<void>;
  }
}

function loadGoogleMaps(key: string): Promise<void> {
  if (typeof google !== "undefined" && google.maps) {
    return Promise.resolve();
  }
  if (!window.__posyMapsLoader) {
    window.__posyMapsLoader = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&v=weekly&loading=async`;
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

// Address inputs for the saveAddress form. With a Google Maps key
// configured, a Places (New) autocomplete search box appears above the
// fields: picking a suggestion fills street/city/state/zip. Apt/unit is
// never autofilled (suggestions don't carry unit numbers) and every
// field stays hand-editable after a pick. Without the key, it's just
// the plain form.
export function AddressFields({ defaults }: { defaults: Defaults }) {
  const searchBoxRef = useRef<HTMLDivElement>(null);
  const line1Ref = useRef<HTMLInputElement>(null);
  const [city, setCity] = useState(defaults.city ?? "");
  const [state, setState] = useState(defaults.state ?? "");
  const [zip, setZip] = useState(defaults.postal_code ?? "");
  const [searchReady, setSearchReady] = useState(false);

  useEffect(() => {
    if (!MAPS_KEY || !searchBoxRef.current) return;
    let element: HTMLElement | undefined;
    let cancelled = false;

    loadGoogleMaps(MAPS_KEY)
      .then(async () => {
        const places = (await google.maps.importLibrary(
          "places",
        )) as google.maps.PlacesLibrary;
        if (cancelled || !searchBoxRef.current) return;

        const autocomplete = new places.PlaceAutocompleteElement({
          includedRegionCodes: ["us"],
        });
        element = autocomplete as unknown as HTMLElement;
        searchBoxRef.current.appendChild(element);
        setSearchReady(true);

        const onSelect = async (event: Event) => {
          const e = event as Event & {
            placePrediction?: { toPlace(): google.maps.places.Place };
            place?: google.maps.places.Place;
          };
          const place = e.placePrediction?.toPlace() ?? e.place;
          if (!place) return;
          await place.fetchFields({ fields: ["addressComponents"] });
          const components = place.addressComponents ?? [];
          const part = (type: string, short = false) => {
            const c = components.find((component) =>
              component.types.includes(type),
            );
            return c ? ((short ? c.shortText : c.longText) ?? "") : "";
          };
          const street = `${part("street_number")} ${part("route")}`.trim();
          if (line1Ref.current && street) line1Ref.current.value = street;
          setCity(
            part("locality") || part("sublocality") || part("postal_town"),
          );
          setState(part("administrative_area_level_1", true));
          setZip(part("postal_code"));
        };

        // Event name differs across Places (New) element versions.
        element.addEventListener("gmp-select", onSelect);
        element.addEventListener("gmp-placeselect", onSelect);
      })
      .catch(() => {
        // Autocomplete is an enhancement; the plain form keeps working.
      });

    return () => {
      cancelled = true;
      element?.remove();
    };
  }, []);

  return (
    <>
      {MAPS_KEY && (
        <label>
          <Text as="div" size="2" mb="1" weight="medium">
            Find the address
          </Text>
          <div ref={searchBoxRef} />
          {searchReady && (
            <Text as="div" size="1" color="gray" mt="1">
              Pick a match to fill the fields below, then add any apt/unit.
            </Text>
          )}
        </label>
      )}
      <label>
        <Text as="div" size="2" mb="1" weight="medium">
          Street address
        </Text>
        <TextField.Root
          ref={line1Ref}
          name="line1"
          required
          placeholder="123 Oak Street"
          defaultValue={defaults.line1 ?? ""}
        />
      </label>
      <label>
        <Text as="div" size="2" mb="1" weight="medium">
          Apt / unit (optional)
        </Text>
        <TextField.Root name="line2" defaultValue={defaults.line2 ?? ""} />
      </label>
      <label>
        <Text as="div" size="2" mb="1" weight="medium">
          City
        </Text>
        <TextField.Root
          name="city"
          required
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />
      </label>
      <Flex gap="3">
        <label style={{ flex: 1 }}>
          <Text as="div" size="2" mb="1" weight="medium">
            State
          </Text>
          <TextField.Root
            name="state"
            value={state}
            onChange={(e) => setState(e.target.value)}
          />
        </label>
        <label style={{ flex: 1 }}>
          <Text as="div" size="2" mb="1" weight="medium">
            ZIP
          </Text>
          <TextField.Root
            name="postal_code"
            value={zip}
            onChange={(e) => setZip(e.target.value)}
          />
        </label>
      </Flex>
    </>
  );
}
