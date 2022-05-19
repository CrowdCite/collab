import { program } from "commander";
import fs from "fs";
import path from "path";
import { latexParser as latex } from "latex-utensils";
import assert from "assert";
import { collectFilesWithExtensions } from "./shared";

interface AuthorStructure {
    name: string;
    email: string;
    address: string;
    affiliation: string[];
}

interface StringContainer {
    strings: string[];
}

interface SectionStructure {
    name: string;
    strings: string[];
    subsections: SectionStructure[];
}

type X0 = number;
type Y0 = number;
type X1 = number;
type Y1 = number;

type BBox = [X0, Y0, X1, Y1];

interface ReferenceStructure {
    sectionName: string;
    paragraphNumber: number;
}

interface FigureStructure {
    caption: string;
    bbox: BBox;
    strings: string[];
    referencesFrom: ReferenceStructure[];
    referencesTo: ReferenceStructure[];
}

interface PdfDocumentStructure {
    title: string;
    author: AuthorStructure[];
    abstract: string;
    sections: SectionStructure[];
    figures: FigureStructure[];
    references: ReferenceStructure[];
    // Strings that couldn't be placed anywhere else in the inferred structure.
    // Generally there should be no useful content in this array; if there is,
    // then there's a problem.
    strings: [];
}

interface CLIOptions {
    verbose: boolean;
    files: string[];
}

function simplifyDocumentStructure(doc: PdfDocumentStructure) {
    const { abstract, author, figures, references, sections, strings, title } = doc;

    function simplifySectionStructure(section: SectionStructure): SectionStructure {
        const { name, strings, subsections } = section;
        return {
            name,
            strings: [strings.join("")],
            subsections: subsections.map(simplifySectionStructure)
        };
    }

    return {
        abstract,
        author,
        figures,
        references,
        sections: sections.map(simplifySectionStructure),
        strings: [strings.join("")],
        title
    };
}

function processArgs(argv: string[]): CLIOptions {
    let files: string[] = [];
    program
        .version("0.1.0")
        .option("-v, --verbose", "Verbose mode.")
        .arguments("<dir|file.tex>")
        .description(
            "process the given `.tex` file or directory of files. Non-`.tex` files will be ignored. Output files are written next to the input files with the extension `.out`."
        )
        .action(async arg => {
            try {
                files.push(
                    ...collectFilesWithExtensions(arg, [".tex"]).filter(
                        file => !file.endsWith(".out.tex")
                    )
                );
            } catch (err: any) {
                console.error(err);
            }
        });
    program.parse(argv);
    const options = program.opts();

    return {
        verbose: !!options.verbose,
        files
    };
}

function sanityCheck(files: string[]) {
    if (files.length === 0) {
        program.help();
    }
}

const SkipCommandName = [
    "newcommand",
    "documentstyle",
    "singlespace",
    "doublespace",
    "epsscale",
    "bigskip",
    "maketitle"
];
const SkipCommandContents = ["plotone"];

let verbose = false;

function serializeNodes(nodes: latex.Node[], sep: string = ""): string {
    const strings = [];
    for (const node of nodes) {
        verbose && strings.push(`${node.kind}[`);
        switch (node.kind) {
            case "text.string": {
                strings.push(node.content);
                break;
            }
            case "command": {
                verbose && strings.push(`${node.name}`);
                if (SkipCommandName.find(name => name === node.name)) {
                    strings.push(`${serializeNodes(node.args, " ")}`);
                    break;
                }
                strings.push(`\\${node.name} `);
                if (!SkipCommandContents.find(name => name === node.name)) {
                    if (node.args.length > 0) {
                        strings.push(`${serializeNodes(node.args, " ")}`);
                    }
                }
                break;
            }
            case "command.text": {
                strings.push(serializeNodes([node.arg]));
                break;
            }
            case "command.def": {
                break;
            }
            case "command.url": {
                strings.push(node.url);
                break;
            }
            case "command.href": {
                strings.push(node.url);
                strings.push("\n");
                strings.push(serializeNodes(node.content));
                break;
            }
            case "command.label": {
                if (node.name === "label") {
                    strings.push(`\n`);
                }
                strings.push(`\\${node.name}{${node.label}}`);
                if (node.name === "label") {
                    strings.push(`\n`);
                }
                break;
            }
            case "env.math.align":
            case "env.math.aligned":
            case "env": {
                strings.push(" ");
                if (/abstract/i.test(node.name)) {
                    strings.push("Abstract\n");
                }
                strings.push(`\n\\begin{${node.name}}\n`);
                strings.push(serializeNodes(node.args));
                strings.push(serializeNodes(node.content));
                strings.push(`\n\\end{${node.name}}\n`);
                break;
            }
            case "arg.group": {
                strings.push("{");
                strings.push(serializeNodes(node.content));
                strings.push("}");
                break;
            }
            case "arg.optional": {
                strings.push(serializeNodes(node.content));
                break;
            }
            case "parbreak": {
                strings.push("\n\n");
                break;
            }
            case "space": {
                strings.push(" ");
                break;
            }
            case "softbreak": {
                strings.push("\n");
                break;
            }
            case "linebreak": {
                strings.push("\n\n");
                break;
            }
            case "superscript": {
                strings.push("^");
                strings.push(serializeNodes(node.arg ? [node.arg] : []));
                break;
            }
            case "subscript": {
                strings.push("_");
                strings.push(serializeNodes(node.arg ? [node.arg] : []));
                break;
            }
            case "alignmentTab": {
                strings.push("&");
                break;
            }
            case "commandParameter": {
                break;
            }
            case "activeCharacter": {
                // ~ is a non-breaking space, not sure if this ast node
                // represents other active characters. We assume space.
                strings.push(" ");
                break;
            }
            case "ignore": {
                break;
            }
            case "verb": {
                // \verb verbatim quoted string
                strings.push(node.content);
                break;
            }
            case "env.verbatim": {
                //  verbatim quoted string
                strings.push(node.content);
                break;
            }
            case "env.minted": {
                // Code highlighting package, basically verbatim
                strings.push(node.content);
                break;
            }
            case "env.lstlisting": {
                // Code highlighting package, basically verbatim
                strings.push(node.content);
                break;
            }
            case "inlineMath": {
                let mathContent = serializeNodes(node.content);
                mathContent = mathContent.trimEnd();
                strings.push(`\$${mathContent}\$`);
                break;
            }
            case "displayMath": {
                strings.push(`\n\\[\n`);
                strings.push(serializeNodes(node.content));
                strings.push(`\n\\]`);
                break;
            }
            case "math.character": {
                strings.push(node.content);
                break;
            }
            case "math.matching_delimiters": {
                strings.push(`\\left${node.left}`);
                strings.push(serializeNodes(node.content));
                strings.push(`\\right${node.right}`);
                break;
            }
            case "math.math_delimiters": {
                if (node.lcommand !== "") {
                    strings.push(`\\${node.lcommand}`);
                }
                strings.push(node.left);
                strings.push(serializeNodes(node.content));
                if (node.rcommand !== "") {
                    strings.push(`\\${node.rcommand}`);
                }
                strings.push(node.right);
                break;
            }
            default:
                throw new Error(`Unknown node kind: ${(node as any).kind}`);
        }
        verbose && strings.push("]");
    }
    return strings.join(sep);
}

function extractDocumentStructure(origNodes: latex.Node[]): PdfDocumentStructure {
    const doc: PdfDocumentStructure = {
        abstract: "",
        sections: [],
        title: "",
        author: [],
        figures: [],
        references: [],
        strings: []
    };

    function helper(nodes: latex.Node[], context: StringContainer) {
        for (const node of nodes) {
            const strings = context.strings;
            switch (node.kind) {
                case "text.string": {
                    strings.push(node.content);
                    break;
                }
                case "command": {
                    if (SkipCommandName.find(name => name === node.name)) {
                        strings.push(`${serializeNodes(node.args, " ")}`);
                        break;
                    }
                    if (node.name === "abstract") {
                        if (doc.abstract) {
                            console.warn(`Duplicate abstract found`);
                        }
                        doc.abstract = serializeNodes(node.args, " ");
                    } else if (node.name === "title") {
                        if (doc.title) {
                            console.warn(`Duplicate title found`);
                        }
                        doc.title = serializeNodes(node.args, " ");
                    } else if (node.name === "author") {
                        doc.author.push({
                            name: serializeNodes(node.args, " "),
                            email: "",
                            address: "",
                            affiliation: []
                        });
                    } else if (node.name === "address") {
                        assert(doc.author.length > 0, `address must follow author`);
                        doc.author[doc.author.length - 1].address = serializeNodes(
                            node.args,
                            " "
                        );
                    } else if (node.name === "section") {
                        const section: SectionStructure = {
                            name: serializeNodes(node.args, " "),
                            strings: [],
                            subsections: []
                        };
                        doc.sections.push(section);
                        context = section;
                    } else if (node.name === "subsection") {
                        const subsection: SectionStructure = {
                            name: serializeNodes(node.args, " "),
                            strings: [],
                            subsections: []
                        };
                        assert(doc.sections.length > 0, `subsection must follow section`);
                        doc.sections[doc.sections.length - 1].subsections.push(
                            subsection
                        );
                        context = subsection;
                    } else {
                        strings.push(`\\${node.name} `);
                        if (!SkipCommandContents.find(name => name === node.name)) {
                            if (node.args.length > 0) {
                                strings.push(`${serializeNodes(node.args, " ")}`);
                            }
                        }
                    }
                    break;
                }
                case "command.text": {
                    strings.push(serializeNodes([node.arg]));
                    break;
                }
                case "command.def": {
                    break;
                }
                case "command.url": {
                    strings.push(node.url);
                    break;
                }
                case "command.href": {
                    strings.push(node.url);
                    strings.push("\n");
                    strings.push(serializeNodes(node.content));
                    break;
                }
                case "command.label": {
                    if (node.name === "label") {
                        strings.push(`\n`);
                    }
                    strings.push(`\\${node.name}{${node.label}}`);
                    if (node.name === "label") {
                        strings.push(`\n`);
                    }
                    break;
                }
                case "env.math.align":
                case "env.math.aligned":
                case "env": {
                    strings.push(" ");
                    if (/abstract/i.test(node.name)) {
                        if (doc.abstract) {
                            console.warn(`Duplicate abstract found`);
                        }
                        doc.abstract = serializeNodes(node.content, " ");
                    } else if (node.name === "document") {
                        helper(node.content, doc);
                    } else {
                        strings.push(`\n\\begin{${node.name}}\n`);
                        strings.push(serializeNodes(node.args));
                        strings.push(serializeNodes(node.content));
                        strings.push(`\n\\end{${node.name}}\n`);
                    }
                    break;
                }
                case "arg.group": {
                    helper(node.content, context);
                    // strings.push("{");
                    // strings.push(serializeNodes(node.content));
                    // strings.push("}");
                    break;
                }
                case "arg.optional": {
                    strings.push(serializeNodes(node.content));
                    break;
                }
                case "parbreak": {
                    strings.push("\n\n");
                    break;
                }
                case "space": {
                    strings.push(" ");
                    break;
                }
                case "softbreak": {
                    strings.push("\n");
                    break;
                }
                case "linebreak": {
                    strings.push("\n\n");
                    break;
                }
                case "superscript": {
                    strings.push("^");
                    strings.push(serializeNodes(node.arg ? [node.arg] : []));
                    break;
                }
                case "subscript": {
                    strings.push("_");
                    strings.push(serializeNodes(node.arg ? [node.arg] : []));
                    break;
                }
                case "alignmentTab": {
                    strings.push("&");
                    break;
                }
                case "commandParameter": {
                    break;
                }
                case "activeCharacter": {
                    // ~ is a non-breaking space, not sure if this ast node
                    // represents other active characters. We assume space.
                    strings.push(" ");
                    break;
                }
                case "ignore": {
                    break;
                }
                case "verb": {
                    // \verb verbatim quoted string
                    strings.push(node.content);
                    break;
                }
                case "env.verbatim": {
                    //  verbatim quoted string
                    strings.push(node.content);
                    break;
                }
                case "env.minted": {
                    // Code highlighting package, basically verbatim
                    strings.push(node.content);
                    break;
                }
                case "env.lstlisting": {
                    // Code highlighting package, basically verbatim
                    strings.push(node.content);
                    break;
                }
                case "inlineMath": {
                    let mathContent = serializeNodes(node.content);
                    mathContent = mathContent.trimEnd();
                    strings.push(`\$${mathContent}\$`);
                    break;
                }
                case "displayMath": {
                    strings.push(`\n\\[\n`);
                    strings.push(serializeNodes(node.content));
                    strings.push(`\n\\]`);
                    break;
                }
                case "math.character": {
                    strings.push(node.content);
                    break;
                }
                case "math.matching_delimiters": {
                    strings.push(`\\left${node.left}`);
                    strings.push(serializeNodes(node.content));
                    strings.push(`\\right${node.right}`);
                    break;
                }
                case "math.math_delimiters": {
                    if (node.lcommand !== "") {
                        strings.push(`\\${node.lcommand}`);
                    }
                    strings.push(node.left);
                    strings.push(serializeNodes(node.content));
                    if (node.rcommand !== "") {
                        strings.push(`\\${node.rcommand}`);
                    }
                    strings.push(node.right);
                    break;
                }
                default:
                    throw new Error(`Unknown node kind: ${(node as any).kind}`);
            }
        }
    }

    helper(origNodes, doc);
    return doc;
}

function cleanup(fileContents: string) {
    let inMath = false;
    let rv = "";
    for (const segment of fileContents.split(/(\$\$)/)) {
        if (segment === "$$") {
            if (inMath) {
                inMath = false;
                // The newline ensures that what comes afterwards isn't parsed
                // as part of the \] command.
                rv += "\\]";
            } else {
                inMath = true;
                rv += "\\[";
            }
        } else {
            rv += segment;
        }
    }
    return rv;
}

function processLatexFiles(opts: CLIOptions) {
    const { files } = opts;
    for (const file of files) {
        console.log(`latexExtract ${file}`);
        const latexContents = fs.readFileSync(file, "utf8");
        const cleanedUpLatex = cleanup(latexContents);
        const parsed = latex.parse(cleanedUpLatex);

        const { dir, name } = path.parse(file);
        const outputFile = `${dir}/${name}.out.tex`;
        fs.writeFileSync(outputFile, serializeNodes(parsed.content));
        console.log(`Wrote ${outputFile}`);

        const outputFileJson = `${dir}/${name}.json`;
        const doc = extractDocumentStructure(parsed.content);
        fs.writeFileSync(outputFileJson, JSON.stringify(simplifyDocumentStructure(doc)));
        console.log(`Wrote ${outputFileJson}`);
    }
}

function main() {
    const opts = processArgs(process.argv);
    sanityCheck(opts.files);
    processLatexFiles(opts);
}

main();
