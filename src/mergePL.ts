import { collectFilesWithExtensions } from "./shared";
import fs from "fs-extra";
import { program } from "commander";

interface CLIOptions {
    dirs: string[];
    outputDir: string;
    verbose: boolean;
}

function processArgs(argv: string[]): CLIOptions {
    let dirs: string[] = [];
    program
        .version("0.1.0")
        .option("-v, --verbose", "Verbose mode.")
        .arguments("<dirs...>")
        .description(
            "Match (2 or more) input directories with .tex and .pdf files and merge them into an output directory."
        )
        .option(`-o, --output <dir>`, "Output directory.")
        .action(async (args: string[]) => {
            try {
                dirs.push(...args);
            } catch (err: any) {
                console.error(err);
            }
        });
    program.parse(argv);
    const options = program.opts();

    return {
        verbose: !!options.verbose,
        dirs,
        outputDir: options.output ?? "merged"
    };
}

import path from "path";

function merge(dirs: string[], outputDir: string) {
    let files: string[] = [];
    for (const dir of dirs) {
        files.push(
            ...collectFilesWithExtensions(dir, [".tex", ".pdf"]).filter(
                file => !file.endsWith(".out.tex")
            )
        );
    }

    const pdfFiles: Map<string, path.ParsedPath> = new Map();
    const latexFiles: Map<string, path.ParsedPath> = new Map();

    for (const file of files) {
        const parsedPath = path.parse(file);
        const { dir, ext, name } = parsedPath;
        if (ext === ".pdf") {
            if (pdfFiles.has(name)) {
                console.warn(`Duplicate pdf file: "${name}"`);
            }
            pdfFiles.set(name, parsedPath);
        } else if (ext === ".tex") {
            const p = dir.split("/");
            let srcIndex = p.findIndex(entry => entry === "src");
            if (srcIndex === -1) {
                srcIndex = p.length - 2;
            }
            const parentDir = p[srcIndex + 1];
            const existing = latexFiles.get(parentDir);
            if (existing) {
                // if (existing.dir !== dir) {
                //     console.warn(
                //         `Duplicate latex file "${name}" for pdf file "${parentDir}"`
                //     );
                // }
            }
            latexFiles.set(parentDir, parsedPath);
        } else {
            console.warn(`Unknown file type: ${file}`);
        }
    }

    const missingLatex: path.ParsedPath[] = [];

    let merged = 0;
    for (const [name, pdfFile] of pdfFiles) {
        // console.log(`Processing pdf: ${name}`);
        const latexFile = latexFiles.get(name);
        if (latexFile) {
            latexFiles.delete(name);
            const destDir = `${outputDir}/${name}`;
            fs.mkdirSync(destDir, { recursive: true });
            fs.createSymlinkSync(
                `${pdfFile.dir}/${pdfFile.base}`,
                `${destDir}/${pdfFile.base}`,
                "file"
            );
            fs.createSymlinkSync(latexFile.dir, `${destDir}/tex`, "dir");
            merged++;
        } else {
            missingLatex.push(pdfFile);
        }
    }

    for (const latexFile of [...latexFiles.values()].slice(0, 10)) {
        console.warn(`Missing pdf for latex file: ${latexFile.dir}/${latexFile.base}`);
    }
    for (const missing of missingLatex.slice(0, 10)) {
        console.warn(`Missing latex file for pdf file: ${missing.dir}/${missing.base}`);
    }

    console.log(`merged ${merged}`);
    console.log(`missing pdf: ${latexFiles.size}`);
    console.log(`missing latex: ${missingLatex.length}`);
}

function main() {
    const opts = processArgs(process.argv);
    console.log(`opts: ${JSON.stringify(opts, undefined, 2)}`);
    const prefix = String(Date.now());
    const finalOutput = `${opts.outputDir}/${prefix}`;
    merge(opts.dirs, finalOutput);
    console.log(`Wrote directory ${finalOutput}`);
}

main();
