<!DOCTYPE html>
<html lang="ko">

<head>
    <meta charset="UTF-8">
    <meta name="description" content="">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <!-- The above 4 meta tags *must* come first in the head; any other head content must come *after* these tags -->

    <!-- Title -->
    <title>Azuma</title>

    <!-- Favicon -->
    <link rel="icon" href="img/core-img/favicon.ico">

    <!-- Stylesheet -->
    <link rel="stylesheet" href="style.css">

</head>

<body>
    <!-- Preloader -->
    <div class="preloader d-flex align-items-center justify-content-center">
        <div class="lds-ellipsis">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
        </div>
    </div>

    <!-- ##### Header Area Start ##### -->
    <header class="header-area">
        <!-- Navbar Area -->
        <div class="oneMusic-main-menu">
            <div class="classy-nav-container breakpoint-off">
                <div class="container">
                    <!-- Menu -->
                    <nav class="classy-navbar justify-content-between" id="oneMusicNav">

                        <!-- Nav brand -->
                        <a href="index.php" class="nav-brand"><img src="img/core-img/logo.png" alt=""></a>

                        <!-- Navbar Toggler -->
                        <div class="classy-navbar-toggler">
                            <span class="navbarToggler"><span></span><span></span><span></span></span>
                        </div>

                        <!-- Menu -->
                        <div class="classy-menu">

                            <!-- Close Button -->
                            <div class="classycloseIcon">
                                <div class="cross-wrap"><span class="top"></span><span class="bottom"></span></div>
                            </div>

                            <!-- Nav Start -->
                            <div class="classynav">
                                <ul>
                                    <li><a href="index.php">Home</a></li>
                                    <li><a href="profile.php">About</a>
                                    <li><a href="blog1.php">Work</a></li>

                                    <!-- Dropdown Menu Start -->
                                    <li><a href="#">Skills</a>
                                      <ul class="dropdown">
                                        <li><a href="blog2.php">Skill Tree</a></li>

                                        <!-- Even Dropdown Start -->
                                        <li><a href="#">Codes</a>
                                          <ul class="dropdown">
                                              <li><a href="subpages/java/spring/bbs1/bbs.php">BBS1:Java</a></li>
                                              <li><a href="subpages/java/spring/bbs2/bbs.php">BBS2:Java??????</a></li>
                                              <li><a href="subpages/java/spring/bbs3/bbs.php">BBS3:??????</a></li>
                                          </ul>
                                        </li>
                                        <!-- Even Dropdown End -->

                                        <?php if(isset($_SESSION['nickname'])){ // ????????? ??? ?????? ?>

                                                <!-- Even Dropdown Start -->
                                                <li><a href="#">Secret</a>
                                                    <ul class="dropdown">
                                                        <!-- Deeply Dropdown Start -->
                                                        <li><a href="subpages/secret/bbs1/bbs.php">????????????</a></li>
                                                        <li><a href="subpages/secret/bbs2/bbs.php">?????????</a></li>
                                                        <li><a href="subpages/secret/bbs3/bbs.php">??????</a></li>
                                                        <li><a href="subpages/secret/bbs4/bbs.php">??????</a></li>
                                                        <li><a href="subpages/secret/bbs5/bbs.php">??????</a></li>
                                                    </ul>
                                                </li>
                                                <!-- Even Dropdown End -->

                                              <?php } else { // ????????? ?????? ?????? ?>
                                                <li><a href="#">Secret</a>
                                                    <ul class="dropdown">
                                                        <!-- Deeply Dropdown Start -->
                                                        <li><a href="#" class="disabled">????????????</a></li>
                                                        <li><a href="#" class="disabled">?????????</a></li>
                                                        <li><a href="#" class="disabled">??????</a></li>
                                                        <li><a href="#" class="disabled">??????</a></li>
                                                        <li><a href="#" class="disabled">??????</a></li>
                                                    </ul>
                                                </li>
                                              <?php
                                            }
                                        ?>

                                      </ul>
                                    </li>
                                    <!-- Dropdown Menu End -->

                                    <li><a href="blog3.php">Blog</a></li>
                                    <li><a href="contact.php">Tel.</a></li>
                                </ul>
                                <!-- Nav End -->

                                <!-- Login/Register & Cart Button -->
                                <div class="login-register-cart-button d-flex align-items-center">
                                    <div class="login-register-btn mr-50">
                                      <?php
                                        // ????????? ??? ??????
                                        if(isset($_SESSION['nickname'])){
                                            echo "<a href='_logout.php' id='loginBtn'>Logout</a>";
                                            // ????????? ?????? ??????
                                          } else {
                                            echo "<a href='login.php' id='loginBtn'>Login</a>";
                                          }
                                      ?>
                                    </div>

                                    <!-- <div class="cart-btn">
                                        <p><span class="icon-shopping-cart"></span> <span class="quantity">1</span></p>
                                    </div> -->
                                </div>

                            </div>
                            <!-- Nav End -->

                        </div>
                    </nav>
                </div>
            </div>
        </div>
    </header>
    <!-- ##### Header Area End ##### -->
