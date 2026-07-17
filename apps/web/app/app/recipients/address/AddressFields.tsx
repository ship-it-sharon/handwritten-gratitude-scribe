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

type Suggestion = {
  id: string;
  main: string;
  secondary: string;
  prediction: google.maps.places.PlacePrediction;
};

// Address inputs for the saveAddress form. The street field itself
// suggests addresses as you type (Places API New, via our own dropdown —
// no separate search box, no opt-in). Picking a suggestion fills
// street/city/state/zip; apt/unit is never autofilled; every field stays
// hand-editable. If Maps fails to load, it's simply a plain form.
export function AddressFields({ defaults }: { defaults: Defaults }) {
  const [line1, setLine1] = useState(defaults.line1 ?? "");
  const [city, setCity] = useState(defaults.city ?? "");
  const [state, setState] = useState(defaults.state ?? "");
  const [zip, setZip] = useState(defaults.postal_code ?? "");
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [mapsReady, setMapsReady] = useState(false);
  const [diagnostic, setDiagnostic] = useState("");

  const sessionTokenRef =
    useRef<google.maps.places.AutocompleteSessionToken | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestQueryRef = useRef("");

  useEffect(() => {
    if (!MAPS_KEY) return;
    let cancelled = false;
    loadGoogleMaps(MAPS_KEY)
      .then(async () => {
        await google.maps.importLibrary("places");
        if (!cancelled) setMapsReady(true);
      })
      .catch((error) => {
        // Autocomplete is an enhancement; the plain form keeps working —
        // but say why, so misconfiguration isn't invisible.
        console.error("Posy address autocomplete:", error);
        if (!cancelled)
          setDiagnostic(
            `Address suggestions unavailable — Google Maps failed to load (${
              error?.message ?? "unknown error"
            }). Check that the key allows this website and the Maps JavaScript API.`,
          );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function onLine1Change(value: string) {
    setLine1(value);
    latestQueryRef.current = value;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!mapsReady || value.trim().length < 3) {
      setSuggestions([]);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      try {
        if (!sessionTokenRef.current) {
          sessionTokenRef.current =
            new google.maps.places.AutocompleteSessionToken();
        }
        const { suggestions: results } =
          await google.maps.places.AutocompleteSuggestion.fetchAutocompleteSuggestions(
            {
              input: value,
              sessionToken: sessionTokenRef.current,
              includedRegionCodes: ["us"],
            },
          );
        // A slower response for an older query must not clobber newer input.
        if (latestQueryRef.current !== value) return;
        setDiagnostic("");
        setSuggestions(
          results
            .map((s) => s.placePrediction)
            .filter((p): p is google.maps.places.PlacePrediction => !!p)
            .slice(0, 5)
            .map((p) => ({
              id: p.placeId,
              main: p.mainText?.text ?? p.text.text,
              secondary: p.secondaryText?.text ?? "",
              prediction: p,
            })),
        );
      } catch (error) {
        console.error("Posy address autocomplete:", error);
        setSuggestions([]);
        setDiagnostic(
          `Address suggestions unavailable — ${
            (error as Error)?.message ?? "lookup failed"
          }`,
        );
      }
    }, 250);
  }

  async function pick(suggestion: Suggestion) {
    setSuggestions([]);
    try {
      const place = suggestion.prediction.toPlace();
      await place.fetchFields({ fields: ["addressComponents"] });
      const components = place.addressComponents ?? [];
      const part = (type: string, short = false) => {
        const c = components.find((component) =>
          component.types.includes(type),
        );
        return c ? ((short ? c.shortText : c.longText) ?? "") : "";
      };
      const street = `${part("street_number")} ${part("route")}`.trim();
      setLine1(street || suggestion.main);
      setCity(part("locality") || part("sublocality") || part("postal_town"));
      setState(part("administrative_area_level_1", true));
      setZip(part("postal_code"));
    } catch {
      setLine1(suggestion.main);
    } finally {
      // A pick ends the billing session; the next keystroke starts fresh.
      sessionTokenRef.current = null;
    }
  }

  return (
    <>
      <label style={{ position: "relative", display: "block" }}>
        <Text as="div" size="2" mb="1" weight="medium">
          Street address
        </Text>
        <TextField.Root
          name="line1"
          required
          placeholder="Start typing the address…"
          value={line1}
          onChange={(e) => onLine1Change(e.target.value)}
          onBlur={() => setTimeout(() => setSuggestions([]), 150)}
          autoComplete="off"
        />
        {suggestions.length > 0 && (
          <div
            style={{
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              zIndex: 10,
              marginTop: 4,
              background: "white",
              border: "1px solid var(--gray-5)",
              borderRadius: "var(--radius-3)",
              boxShadow: "var(--shadow-4)",
              overflow: "hidden",
            }}
          >
            {suggestions.map((s) => (
              <button
                key={s.id}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(s);
                }}
                style={{
                  display: "block",
                  width: "100%",
                  textAlign: "left",
                  padding: "0.5rem 0.75rem",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  font: "inherit",
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLElement).style.background =
                    "var(--gray-3)";
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLElement).style.background = "none";
                }}
              >
                <Text size="2">{s.main}</Text>{" "}
                <Text size="1" color="gray">
                  {s.secondary}
                </Text>
              </button>
            ))}
          </div>
        )}
        {diagnostic && (
          <Text as="div" size="1" color="amber" mt="1">
            {diagnostic}
          </Text>
        )}
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
