import Link from "next/link";
import { Button, Card, Flex, Text } from "@radix-ui/themes";

export default function LandingPage() {
  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <Text as="p" size="4" color="gray" mt="2">
        Thank-you notes that sound like you, look like your handwriting, and
        arrive in the mail — without the burden.
      </Text>

      <Card size="3" mt="5">
        <Flex direction="column" gap="3">
          <Text as="p">
            Weddings, showers, birthdays: the gratitude is real, the stack of
            blank cards is daunting. Posy helps you write each note in your
            own voice, renders it in a handwriting style that&rsquo;s yours,
            and prints and mails every card — matching envelope included.
          </Text>
          <Text as="p" size="2" color="gray">
            This is the walking skeleton of Posy&rsquo;s V1. The real landing
            page arrives with milestone M5.
          </Text>
          <Flex>
            <Button asChild size="3">
              <Link href="/login">Sign in</Link>
            </Button>
          </Flex>
        </Flex>
      </Card>
    </main>
  );
}
