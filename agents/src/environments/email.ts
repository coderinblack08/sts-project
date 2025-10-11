import { tool } from "ai";
import { z } from "zod";
import type { ToolWithPolicies } from "../agents/tool.ts";

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
