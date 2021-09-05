import React from "react";
import classes from "./Layout.module.scss";

import Header from '../../components/Header/Header';
import Main from '../../components/Main/Main'
import Footer from '../../components/Footer/Footer'

function Layout() {
    return (
        <div className={classes.Layout}>
            <Header></Header>
            <Main></Main>
            <Footer></Footer>
        </div>
    );
}
export default Layout