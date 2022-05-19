import fs from "fs";
import path from "path";

export type DirectoryOrFilename = string;

export function hasExtension(file: string, ext: string | string[]): boolean {
    if (typeof ext === "string") {
        ext = [ext];
    }
    const rv = ext.find(e => path.extname(file) === e) !== undefined;
    return rv;
}

export function filesInDirectory(dir: string) {
    return fs.readdirSync(dir).map(file => path.join(dir, file));
}

export function collectFilesWithExtensions(
    arg: DirectoryOrFilename | undefined,
    exts: string[]
): string[] {
    if (!arg) {
        return [];
    }
    const stats = fs.statSync(arg);
    let fileList: string[] = [];
    if (stats.isFile() && hasExtension(arg, exts)) {
        fileList = [arg];
    } else if (stats.isDirectory()) {
        for (const entry of filesInDirectory(arg)) {
            fileList.push(...collectFilesWithExtensions(entry, exts));
        }
    }
    return fileList;
}
