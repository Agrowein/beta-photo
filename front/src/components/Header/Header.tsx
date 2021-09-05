import React from "react";

import classes from "./Header.module.scss";
import Navigation from "../Navigation/Navigation"

function Header() {
    return (
        <header className={classes.Header}>
            <span className={classes.logo}>Better Photo</span>
            <Navigation></Navigation>
        </header>
    );
}

export default Header