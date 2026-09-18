// run_primitive.ts — execute one JevML primitive on a canned workload.
// usage: node --experimental-strip-types run_primitive.ts <primitive-id>
import { pca, generateElongated } from "./src/pca.ts";
import { metropolisHastings, getTarget } from "./src/mcmc.ts";
import { runTextDiffusion } from "./src/text-diffusion.ts";
import { createSim, stepSim } from "./src/nca.ts";

const id = process.argv[2];

function asciiHist(xs: number[], bins = 22) {
  const lo = Math.min(...xs), hi = Math.max(...xs), w = (hi - lo) / bins || 1;
  const c = new Array(bins).fill(0);
  for (const x of xs) c[Math.min(bins - 1, Math.floor((x - lo) / w))]++;
  const m = Math.max(...c);
  for (let i = 0; i < bins; i++)
    console.log(`  ${(lo + i * w).toFixed(2).padStart(7)} |${"#".repeat(Math.round(c[i] / m * 44))}`);
}

function asciiGrid(flat: ArrayLike<number>, w: number, h: number) {
  for (let y = 0; y < h; y++) {
    let line = "  ";
    for (let x = 0; x < w; x++) {
      const v = flat[y * w + x];
      line += v > 0.5 ? "#" : v > 0.12 ? "." : " ";
    }
    console.log(line);
  }
}

if (id === "pca") {
  const X = generateElongated({ n: 60, seed: 7, stretch: 6 });
  const r = pca(X, 2);
  console.log("explained variance:", r.explainedVarianceRatio.map(v => (v * 100).toFixed(1) + "%").join("  "));
  console.log("PC1:", r.components[0].map(v => v.toFixed(2)).join("  "), " PC2:", r.components[1].map(v => v.toFixed(2)).join("  "));
  console.log("\nprojected onto PC1 — the table's true axis:");
  asciiHist(r.scores.map(s => s[0]));
} else if (id === "mcmc") {
  const t = getTarget("wells");
  const r = metropolisHastings({ target: t, start: [0], stepSize: 0.9, nSamples: 4000, seed: 3 });
  const xs = r.samples.map(s => s[0]);
  console.log(`acceptance rate: ${(r.acceptanceRate * 100).toFixed(1)}%   samples: ${xs.length}`);
  console.log("\nthe chain found both wells — bimodal histogram:");
  asciiHist(xs);
} else if (id === "text-diffusion") {
  const clean = "deploy the patch once the cluster is green";
  console.log(`target    : ${clean}`);
  const r = runTextDiffusion({ target: clean, mode: "restore", steps: 10, seed: 11 });
  for (const s of r) console.log(`  t=${s.t.toFixed(2)}: ${s.text}`);
  console.log(`restored  : ${r.at(-1)?.text}`);
} else if (id === "nca") {
  const s = createSim("life", 40, 18, 5);
  const show = (tag: string, n: number) => {
    console.log(`\nt=${n}:`);
    asciiGrid(s.cells, s.width, s.height);
  };
  console.log("t=0 (random seed — Conway's Life):");
  asciiGrid(s.cells, s.width, s.height);
  stepSim(s, 15); show("", 15);
  stepSim(s, 35); show("", 50);
} else {
  console.error("unknown primitive:", id);
  process.exit(1);
}
