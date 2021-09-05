import React from "react";

import classes from "./Navigation.module.scss";
import Button from '../Button/Button'

function Navigation() {
    const loginButtonHandler = () => {
        alert("Clecked 'Login'")
    }

    const singinButtonHandler = () => {
        alert("Clicked 'Sing In'")
    }

    return (
        <nav className={classes.Navigation}>
            <ul>
                <li><Button class={classes.button} title="Login" cb={loginButtonHandler}></Button></li>
                <li><Button class={classes.button} title="Sing In" cb={singinButtonHandler}></Button></li>    
            </ul>
        </nav>
    );
}
export default Navigation