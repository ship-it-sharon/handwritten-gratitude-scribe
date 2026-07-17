import Link from "next/link";
import { redirect } from "next/navigation";
import {
  Badge,
  Button,
  Card,
  Flex,
  Heading,
  Separator,
  Text,
} from "@radix-ui/themes";
import { isSupabaseConfigured } from "../../lib/supabase/config";
import { createClient } from "../../lib/supabase/server";

const OCCASION_LABELS: Record<string, string> = {
  wedding: "Wedding",
  bridal_shower: "Bridal shower",
  baby_shower: "Baby shower",
  birthday: "Birthday",
  graduation: "Graduation",
  holiday: "Holiday",
  other: "Occasion",
};

export default async function AppHome() {
  if (!isSupabaseConfigured) {
    redirect("/login");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { data: events } = await supabase
    .from("events")
    .select("id, title, occasion_type, event_date, event_recipients (id)")
    .order("created_at", { ascending: false });

  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <Text as="p" size="2" color="gray">
        {user.email}
      </Text>

      <Card size="3" mt="5">
        <Flex direction="column" gap="4">
          <Heading size="5">Your events</Heading>
          {!events || events.length === 0 ? (
            <Text as="p" color="gray">
              Nothing here yet. An event is the occasion you&rsquo;re saying
              thank you for — a wedding, a shower, a birthday.
            </Text>
          ) : (
            <Flex direction="column">
              {events.map((event, i) => (
                <Flex key={event.id} direction="column">
                  {i > 0 && <Separator size="4" my="3" />}
                  <Flex justify="between" align="center" gap="3">
                    <Flex direction="column">
                      <Link href={`/app/events/${event.id}`}>
                        <Text weight="medium">{event.title}</Text>
                      </Link>
                      <Text size="2" color="gray">
                        {OCCASION_LABELS[event.occasion_type] ?? "Occasion"}
                        {event.event_date ? ` · ${event.event_date}` : ""}
                      </Text>
                    </Flex>
                    <Badge variant="soft">
                      {(event.event_recipients ?? []).length} recipient
                      {(event.event_recipients ?? []).length === 1 ? "" : "s"}
                    </Badge>
                  </Flex>
                </Flex>
              ))}
            </Flex>
          )}
          <Flex>
            <Button asChild>
              <Link href="/app/events/new">Create an event</Link>
            </Button>
          </Flex>
        </Flex>
      </Card>

      <Card size="2" mt="4">
        <form action="/auth/signout" method="post">
          <Button variant="soft" color="gray" type="submit">
            Sign out
          </Button>
        </form>
      </Card>
    </main>
  );
}
