<?php

use OpenSwoole\Server;
use Phpml\Tokenization\WhitespaceTokenizer;

require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/EnjoyNaiveBayes.php';

/**
 * ===========================
 * 1. Загружаем модель ОДИН РАЗ
 * ===========================
 */
$modelPath = __DIR__ . '/storage/category_model.php';

if (!file_exists($modelPath)) {
    die("❌ Модель не найдена\n");
}

$data = unserialize(file_get_contents($modelPath));

if (!isset($data['vocab'], $data['nb'])) {
    die("❌ Неверный формат модели\n");
}

$vocab = $data['vocab'];
$nb    = $data['nb'];

$tokenizer = new WhitespaceTokenizer();

echo "✅ Модель загружена\n";

/**
 * ===========================
 * 2. Unix socket
 * ===========================
 */
$socketPath = '/tmp/phpml.sock';

// если сокет уже существует — удаляем
if (file_exists($socketPath)) {
    unlink($socketPath);
}

/*
Server::__construct(
    string $host,
    int    $port,
    int    $mode,
    int    $sockType
)
*/

$server = new Server(
    $socketPath,
    0,
    SWOOLE_PROCESS,
    SWOOLE_SOCK_UNIX_STREAM
);

$server->set([
    'worker_num' => 1,
    'max_request' => 0,
]);

/**
 * ===========================
 * 3. Токенизация
 * ===========================
 */
function tokensToSparseVector(string $text, array $vocab, $tokenizer): array {
    $vec = [];
    $tokens = $tokenizer->tokenize(mb_strtolower(trim($text)));
    foreach ($tokens as $token) {
        if (isset($vocab[$token])) {
            $idx = $vocab[$token];
            $vec[$idx] = ($vec[$idx] ?? 0) + 1;
        }
    }
    return $vec;
}

/**
 * ===========================
 * 4. Обработка запросов
 * ===========================
 */
$server->on('Receive', function (
    Server $server,
    int $fd,
    int $reactorId,
    string $data
) use ($nb, $vocab, $tokenizer) {

    $payload = json_decode(trim($data), true);

    if (!isset($payload['text'])) {
        $server->send($fd, json_encode([
                'error' => 'Missing field: text'
            ]) . "\n");
        return;
    }

    $vec = tokensToSparseVector($payload['text'], $vocab, $tokenizer);

    $pred = $nb->predictTopN([$vec], 24);

    $server->send($fd, json_encode([
            'top24' => $pred[0],
        ], JSON_UNESCAPED_UNICODE) . "\n");
});

echo "🚀 Swoole UDS сервер запущен: {$socketPath}\n";
$server->start();
