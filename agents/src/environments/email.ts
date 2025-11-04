import { tool } from "ai";
import { z } from "zod";
import { Agent } from "../agents/agent.ts";
import type { ToolWithPolicies } from "../agents/tool.ts";
import { pLLM, qLLM } from "../models/models.ts";

export interface Email {
  id: string;
  from: string;
  to: string;
  subject: string;
  body: string;
  timestamp: Date;
}

export interface InboxData {
  emails: Email[];
  sentEmails: Email[];
}

export class EmailEnvironment {
  constructor(public data: InboxData) {}

  get tools(): Record<string, ToolWithPolicies> {
    return {
      readInbox: {
        policies: [],
        isExternal: true,
        tool: tool({
          description:
            "Read emails from the inbox. Optionally filter by a search term that matches subject, sender, or body.",
          inputSchema: z.object({
            searchTerm: z
              .string()
              .optional()
              .describe("Optional grep-like search term to filter emails"),
          }),
          execute: async (input) => {
            let emails = this.data.emails;

            if (input.searchTerm) {
              const lowerSearch = input.searchTerm.toLowerCase();
              emails = emails.filter(
                (email) =>
                  email.subject.toLowerCase().includes(lowerSearch) ||
                  email.from.toLowerCase().includes(lowerSearch) ||
                  email.body.toLowerCase().includes(lowerSearch)
              );
            }

            return {
              emails: emails.map((email) => ({
                id: email.id,
                from: email.from,
                subject: email.subject,
                timestamp: email.timestamp,
              })),
              count: emails.length,
            };
          },
        }),
      },

      readEmail: {
        policies: [],
        isExternal: true,
        tool: tool({
          description: "Read a specific email by its ID",
          inputSchema: z.object({
            emailId: z.string().describe("The ID of the email to read"),
          }),
          execute: async (input) => {
            const email = this.data.emails.find((e) => e.id === input.emailId);

            if (!email) {
              return { error: "Email not found" };
            }

            return {
              id: email.id,
              from: email.from,
              to: email.to,
              subject: email.subject,
              body: email.body,
              timestamp: email.timestamp,
            };
          },
        }),
      },

      sendEmail: {
        policies: [],
        isExternal: false,
        tool: tool({
          description: "Send an email to a recipient",
          inputSchema: z.object({
            to: z.string().describe("The recipient's email address"),
            subject: z.string().describe("The email subject"),
            body: z.string().describe("The email body"),
          }),
          execute: async (input) => {
            const newEmail: Email = {
              id: `sent-${Date.now()}`,
              from: "user@example.com",
              to: input.to,
              subject: input.subject,
              body: input.body,
              timestamp: new Date(),
            };

            this.data.sentEmails.push(newEmail);

            return {
              success: true,
              message: `Email sent to ${input.to}`,
              emailId: newEmail.id,
            };
          },
        }),
      },
    };
  }
}

const emailData = {
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

const emailEnv = new EmailEnvironment(emailData);
const agent = new Agent(qLLM, pLLM, emailEnv.tools);

const prompt =
  "Summarize my emails and send an email to user@example.com with the summary";

console.time("generate");
const result = await agent.generate(prompt);
console.timeEnd("generate");

console.log("[❓] Prompt:", prompt);
console.log("[📧] Sent emails:", emailEnv.data.sentEmails);
