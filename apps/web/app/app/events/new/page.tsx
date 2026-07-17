import Link from "next/link";
import {
  Callout,
  Card,
  Flex,
  Select,
  Text,
  TextField,
} from "@radix-ui/themes";
import { createEvent } from "../actions";
import { SubmitButton } from "../../../components/SubmitButton";

const OCCASIONS = [
  ["wedding", "Wedding"],
  ["bridal_shower", "Bridal shower"],
  ["baby_shower", "Baby shower"],
  ["birthday", "Birthday"],
  ["graduation", "Graduation"],
  ["holiday", "Holiday"],
  ["other", "Something else"],
] as const;

export default async function NewEventPage({
  searchParams,
}: {
  searchParams: Promise<{
    error?: string;
    message?: string;
    title?: string;
    occasion_type?: string;
    event_date?: string;
  }>;
}) {
  const params = await searchParams;

  return (
    <main className="page">
      <Text as="p" size="2">
        <Link href="/app">&larr; Back</Link>
      </Text>
      <h1 className="wordmark">New event</h1>
      <Card size="3" mt="5">
        <Flex direction="column" gap="4">
          {params.error && (
            <Callout.Root color="red">
              <Callout.Text>
                {params.error === "title"
                  ? "Give your event a name so you can find it again."
                  : `The save didn't go through${
                      params.message
                        ? ` — the database said: “${params.message}”`
                        : ""
                    }. Your entries are still filled in below; try again.`}
              </Callout.Text>
            </Callout.Root>
          )}
          <form action={createEvent}>
            <Flex direction="column" gap="4">
              <label>
                <Text as="div" size="2" mb="1" weight="medium">
                  What&rsquo;s the occasion called?
                </Text>
                <TextField.Root
                  name="title"
                  required
                  size="3"
                  placeholder="Sarah &amp; Tom&rsquo;s Wedding"
                  defaultValue={params.title ?? ""}
                />
              </label>
              <label>
                <Text as="div" size="2" mb="1" weight="medium">
                  What kind of occasion?
                </Text>
                <Select.Root
                  name="occasion_type"
                  defaultValue={params.occasion_type ?? "wedding"}
                  size="3"
                >
                  <Select.Trigger style={{ width: "100%" }} />
                  <Select.Content>
                    {OCCASIONS.map(([value, label]) => (
                      <Select.Item key={value} value={value}>
                        {label}
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Root>
              </label>
              <label>
                <Text as="div" size="2" mb="1" weight="medium">
                  When was it? (optional)
                </Text>
                <TextField.Root
                  type="date"
                  name="event_date"
                  size="3"
                  defaultValue={params.event_date ?? ""}
                />
              </label>
              <Flex>
                <SubmitButton pendingLabel="Creating…" size="3">
                  Create event
                </SubmitButton>
              </Flex>
            </Flex>
          </form>
        </Flex>
      </Card>
    </main>
  );
}
