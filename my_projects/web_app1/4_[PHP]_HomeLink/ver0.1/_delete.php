<?php
$no = $_GET['no'];
require ('_conn.php');
$sql = "UPDATE `home` SET `del_flg` = 1 WHERE `no` = $no";
$result = mysqli_query($conn, $sql);
if ($result) {
    echo "<script>location.href='index.php'</script>";
}
?>
