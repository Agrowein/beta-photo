import React, { DragEvent } from "react";

import classes from './Main.module.scss'
import { FileDrop } from 'react-file-drop'

function Main() {
    const dropFileEvent = (files: FileList | null, event: DragEvent) => {
        console.log(files);
    }
    return (
        <main className={classes.Main}>
            <FileDrop 
                className={classes.dropzone}
                onDrop={dropFileEvent}
            >DropZone</FileDrop>
        </main>
    );
}

export default Main