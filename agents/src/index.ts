import { runDsEval } from "./evals/eval.ts";

const args = Bun.argv;
const n = args[2] ? parseInt(args[2]) : undefined;

const results = await runDsEval(n);
await Bun.write("results.json", JSON.stringify(results, null, 2));
console.log(`[💾] Saved results to results.json`);

// import { Agent } from "./agents/agent.ts";
// import { EmailEnvironment } from "./environments/email.ts";
// import { privilegedModel, quarantinedModel } from "./models/models.ts";

// const emailData = {
//   emails: [
//     {
//       id: "1",
//       from: "john@company.com",
//       to: "user@example.com",
//       subject: "Q4 Sales Report",
//       body: "Hi team, attached is the Q4 sales report. Revenue increased by 25% compared to Q3. Great work!",
//       timestamp: new Date("2024-10-08T10:30:00"),
//     },
//     {
//       id: "2",
//       from: "sarah@company.com",
//       to: "user@example.com",
//       subject: "Meeting Tomorrow",
//       body: "Don't forget about our 2pm meeting tomorrow to discuss the product roadmap.",
//       timestamp: new Date("2024-10-09T14:20:00"),
//     },
//     {
//       id: "3",
//       from: "mike@vendor.com",
//       to: "user@example.com",
//       subject: "Invoice #12345",
//       body: "Please find attached invoice #12345 for last month's services totaling $5,000.",
//       timestamp: new Date("2024-10-10T09:15:00"),
//     },
//   ],
//   sentEmails: [],
// };

// const emailEnv = new EmailEnvironment(emailData);
// const agent = new Agent(quarantinedModel, privilegedModel, emailEnv.tools);

// const prompt =
//   "Summarize my emails and send an email to user@example.com with the summary";

// console.time("generate");
// const result = await agent.generate(prompt);
// console.timeEnd("generate");

// console.log("[❓] Prompt:", prompt);
// console.log("[📧] Sent emails:", emailEnv.data.sentEmails);
