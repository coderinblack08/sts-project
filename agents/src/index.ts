import { runDsEval } from "./evals/eval.ts";

const args = Bun.argv;
const n =
  args[2] === "max" ? undefined : args[2] ? parseInt(args[2]) : undefined;
const batchSize = args[3] ? parseInt(args[3]) : 25;

const results = await runDsEval(n, batchSize);
await Bun.write("results.json", JSON.stringify(results, null, 2));
console.log(`[💾] Saved results to results.json`);
