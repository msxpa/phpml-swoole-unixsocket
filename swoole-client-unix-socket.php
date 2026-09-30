<?php

$socket = stream_socket_client(
    "unix:///tmp/phpml.sock",
    $errno,
    $errstr,
    1
);

if (!$socket) {
    die("Ошибка подключения: $errstr");
}

fwrite($socket, json_encode([
        'text' => $argv[1]
    ]) . "\n");

$response = fgets($socket);
fclose($socket);

var_dump(json_decode($response, true));
