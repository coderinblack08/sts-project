import { runDsEvalBaseline } from "./evals/eval_baseline.ts";

const args = Bun.argv;
const n = args[2] ? parseInt(args[2]) : undefined;
const batchSize = args[3] ? parseInt(args[3]) : 25;

const results = await runDsEvalBaseline(n, batchSize);
await Bun.write("results_baseline.json", JSON.stringify(results, null, 2));
console.log(`[💾] Saved results to results_baseline.json`);

