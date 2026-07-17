import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { Button, Card, Flex, Heading, Text } from "@radix-ui/themes";
import { createClient } from "../../../../../lib/supabase/server";
import { ImportCsv } from "./ImportCsv";

export default async function ImportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: event } = await supabase
    .from("events")
    .select("id, title")
    .eq("id", id)
    .single();
  if (!event) notFound();

  return (
    <main className="page">
      <Text as="p" size="2">
        <Link href={`/app/events/${id}`}>&larr; Back to {event.title}</Link>
      </Text>
      <h1 className="wordmark">Import your list</h1>
      <Card size="3" mt="5">
        <Flex direction="column" gap="4">
          <Text as="p" size="2" color="gray">
            Upload the spreadsheet you already have — a wedding guest list, a
            shower invite list, an export from Zola or The Knot. Names are
            enough; addresses come along if they&rsquo;re there.
          </Text>
          <ImportCsv eventId={id} />
        </Flex>
      </Card>

      <Card size="3" mt="4">
        <Flex direction="column" gap="3">
          <Heading size="5">What the file should look like</Heading>
          <Text as="p" size="2" color="gray">
            Any CSV works as long as there&rsquo;s a header row and a name
            column — we&rsquo;ll figure out the rest and let you correct the
            mapping before anything imports. Columns we understand:
          </Text>
          <ul style={{ paddingLeft: "1.2rem", margin: 0 }}>
            <li>
              <Text size="2" color="gray">
                <b>Name</b> — required; one person or family per row
              </Text>
            </li>
            <li>
              <Text size="2" color="gray">
                <b>Street Address</b> and <b>Apt/Unit</b> — optional
              </Text>
            </li>
            <li>
              <Text size="2" color="gray">
                <b>City</b>, <b>State</b>, <b>Zip</b> — optional
              </Text>
            </li>
          </ul>
          <Text as="p" size="2" color="gray">
            Rows with just a name are fine — addresses can be added any time
            before mailing. Extra columns are simply skipped.
          </Text>
          <Flex>
            <Button asChild variant="soft">
              <a href="/posy-import-template.csv" download>
                Download a template
              </a>
            </Button>
          </Flex>
          <Text as="p" size="2" color="gray">
            Using Google Sheets or Excel? File → Download (or Save As) → CSV.
          </Text>
        </Flex>
      </Card>
    </main>
  );
}
