import React from "react";

interface IButton {
    title: string,
    class: string,
    cb: () => void
}

function Button(props: IButton) {
    return (
        <button className={props.class} onClick={props.cb}>
            {props.title}
        </button>
    );
}

export default Button