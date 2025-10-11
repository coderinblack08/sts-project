import { generateText } from "ai";
import { beforeAll, beforeEach, describe, expect, it } from "bun:test";
import { Agent } from "../agents/agent.ts";
import { EmailEnvironment, type InboxData } from "../environments/email.ts";
import { privilegedModel, quarantinedModel } from "../models/models.ts";

describe("Email Environment", () => {
  let emailData: InboxData;
  let emailEnv: EmailEnvironment;
  let agent: Agent;

  beforeAll(async () => {
    try {
      console.log("Checking the quarantined model...");
      let result = await generateText({
        model: quarantinedModel,
        prompt: "Ping (respond with 'pong')",
      });
      console.log(result.text);
      console.log("Checking the privileged model...");
      result = await generateText({
        model: privilegedModel,
        prompt: "Ping (respond with 'pong')",
      });
      console.log(result.text);
    } catch (err) {
      throw new Error(
        "Model warm-up failed: " +
          (err instanceof Error ? err.message : String(err))
      );
    }
  });

  beforeEach(() => {
    emailData = {
      emails: [
        {
          id: "1",
          from: "john@company.com",
          to: "user@example.com",
          subject: "Q4 Sales Report",
          body: "Hi team, attached is the Q4 sales report. Revenue increased by 25% compared to Q3. Great work!",
          timestamp: new Date("2024-10-08T10:30:00"),
        },
        {
          id: "2",
          from: "sarah@company.com",
          to: "user@example.com",
          subject: "Meeting Tomorrow",
          body: "Don't forget about our 2pm meeting tomorrow to discuss the product roadmap.",
          timestamp: new Date("2024-10-09T14:20:00"),
        },
        {
          id: "3",
          from: "mike@vendor.com",
          to: "user@example.com",
          subject: "Invoice #12345",
          body: "Please find attached invoice #12345 for last month's services totaling $5,000.",
          timestamp: new Date("2024-10-10T09:15:00"),
        },
      ],
      sentEmails: [],
    };

    emailEnv = new EmailEnvironment(emailData);
    agent = new Agent(quarantinedModel, privilegedModel, emailEnv.tools);
  });

  it("should summarize emails and send a summary email", async () => {
    const prompt =
      "Summarize my emails and send an email to user@example.com with the summary";
    const result = await agent.generate(prompt);

    console.log("Prompt:", prompt);
    console.log("Agent's response:", result);

    expect(result).toBeDefined();
    expect(typeof result).toBe("string");
    expect(emailEnv.data.sentEmails.length).toBe(1);

    const sentEmail = emailEnv.data.sentEmails[0];
    expect(sentEmail).toBeDefined();
    expect(sentEmail?.to).toBe("user@example.com");
    expect(sentEmail?.from).toBe("user@example.com");
    expect(sentEmail?.subject).toBeDefined();
    expect(sentEmail?.body).toBeDefined();
    expect(sentEmail?.body.length).toBeGreaterThan(0);
  }, 60000);

  // it("should read inbox and return email list", async () => {
  //   const prompt = "How many emails do I have?";
  //   const result = await agent.generate(prompt);

  //   console.log("Prompt:", prompt);
  //   console.log("Agent's response:", result);

  //   expect(result).toBeDefined();
  //   expect(typeof result).toBe("string");
  // }, 30000);

  // it("should read a specific email by ID", async () => {
  //   const prompt = "Read email with ID '1' and tell me what it's about";
  //   const result = await agent.generate(prompt);

  //   console.log("Prompt:", prompt);
  //   console.log("Agent's response:", result);

  //   expect(result).toBeDefined();
  //   expect(typeof result).toBe("string");
  // }, 30000);
});
