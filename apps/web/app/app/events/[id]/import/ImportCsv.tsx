"use client";

import { useState } from "react";
import Papa from "papaparse";
import { useRouter } from "next/navigation";
import { importCsvRecipients } from "../../actions";

const FIELDS = [
  ["full_name", "Name"],
  ["line1", "Street address"],
  ["line2", "Apt / unit"],
  ["city", "City"],
  ["state", "State"],
  ["postal_code", "ZIP"],
  ["", "— skip this column —"],
] as const;

type FieldKey = (typeof FIELDS)[number][0];

// Guess which app field a CSV header refers to.
function guessField(header: string): FieldKey {
  const h = header.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (/(fullname|^name$|guestname|recipient)/.test(h)) return "full_name";
  if (/(address2|addr2|line2|apt|unit|suite)/.test(h)) return "line2";
  if (/(address|addr|street|line1)/.test(h)) return "line1";
  if (/city|town/.test(h)) return "city";
  if (/^(state|province|region)$/.test(h)) return "state";
  if (/(zip|postal)/.test(h)) return "postal_code";
  return "";
}

export function ImportCsv({ eventId }: { eventId: string }) {
  const router = useRouter();
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<string[][]>([]);
  const [mapping, setMapping] = useState<FieldKey[]>([]);
  const [status, setStatus] = useState<"idle" | "importing" | "error">("idle");
  const [message, setMessage] = useState("");

  function onFile(file: File) {
    Papa.parse<string[]>(file, {
      skipEmptyLines: true,
      complete: (result) => {
        const data = result.data as string[][];
        if (data.length < 2) {
          setStatus("error");
          setMessage(
            "That file needs a header row plus at least one person.",
          );
          return;
        }
        setStatus("idle");
        setMessage("");
        setHeaders(data[0]);
        setRows(data.slice(1));
        setMapping(data[0].map(guessField));
      },
      error: () => {
        setStatus("error");
        setMessage("Couldn't read that file — is it a CSV?");
      },
    });
  }

  async function runImport() {
    if (!mapping.includes("full_name")) {
      setStatus("error");
      setMessage("Point at least one column at “Name” so each row has a person.");
      return;
    }
    setStatus("importing");
    const mapped = rows.map((row) => {
      const record: Record<string, string> = {};
      mapping.forEach((field, i) => {
        if (field && row[i] && !record[field]) record[field] = row[i];
      });
      return record as {
        full_name: string;
        line1?: string;
        line2?: string;
        city?: string;
        state?: string;
        postal_code?: string;
      };
    });
    try {
      const { imported, skipped } = await importCsvRecipients(eventId, mapped);
      router.push(
        `/app/events/${eventId}?imported=${imported}&skipped=${skipped}`,
      );
    } catch {
      setStatus("error");
      setMessage("Import failed partway — check the recipient list and retry.");
    }
  }

  return (
    <div className="stack">
      {headers.length === 0 ? (
        <label className="stack">
          <span>Choose your spreadsheet (CSV file)</span>
          <input
            className="input"
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFile(file);
            }}
          />
        </label>
      ) : (
        <>
          <p className="muted">
            {rows.length} people found. Check that each column points at the
            right thing — we took a guess:
          </p>
          <div className="import-table-wrap">
            <table className="import-table">
              <thead>
                <tr>
                  {headers.map((header, i) => (
                    <th key={i}>
                      <div className="muted">{header}</div>
                      <select
                        className="input"
                        value={mapping[i]}
                        onChange={(e) => {
                          const next = [...mapping];
                          next[i] = e.target.value as FieldKey;
                          setMapping(next);
                        }}
                      >
                        {FIELDS.map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 3).map((row, ri) => (
                  <tr key={ri}>
                    {headers.map((_, ci) => (
                      <td key={ci}>{row[ci]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows.length > 3 && (
            <p className="muted">…and {rows.length - 3} more.</p>
          )}
          <div>
            <button
              className="button"
              onClick={runImport}
              disabled={status === "importing"}
            >
              {status === "importing"
                ? `Importing ${rows.length} people…`
                : `Import ${rows.length} people`}
            </button>
          </div>
        </>
      )}
      {status === "error" && <p className="notice">{message}</p>}
    </div>
  );
}
