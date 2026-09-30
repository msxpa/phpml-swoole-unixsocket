<?php

use OpenSwoole\Http\Server;
use OpenSwoole\Http\Request;
use OpenSwoole\Http\Response;
use Phpml\Tokenization\WhitespaceTokenizer;

require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/EnjoyNaiveBayes.php';

// ---------------------------
// 1. Загружаем модель ОДИН РАЗ
// ---------------------------
$modelPath = __DIR__ . '/storage/category_model.php';

if (!file_exists($modelPath)) {
    die("❌ Модель не найдена: $modelPath\n");
}

echo "📦 Загружаем модель...\n";

$data = unserialize(file_get_contents($modelPath));

if (!is_array($data) || !isset($data['vocab'], $data['nb'])) {
    die("❌ Ошибка: неверный формат модели\n");
}

$vocab = $data['vocab'];   // token → index
$nb    = $data['nb'];      // Naive Bayes classifier

$tokenizer = new WhitespaceTokenizer();

echo "✅ Модель загружена: токенов " . count($vocab) . "\n";
echo "🚀 Swoole сервер запускается...\n";

// ---------------------------
// 2. Функция токенизации как при обучении
// ---------------------------
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

// ---------------------------
// 3. HTTP сервер
// ---------------------------
$server = new Server("0.0.0.0", 9501);

$server->on("Request", function(Request $req, Response $res) use ($nb, $vocab, $tokenizer) {

    if ($req->server['request_method'] === 'OPTIONS') {
        $res->header("Access-Control-Allow-Origin", "*");
        $res->header("Access-Control-Allow-Methods", "POST, OPTIONS");
        $res->header("Access-Control-Allow-Headers", "Content-Type");
        $res->status(200);
        $res->end();
        return;
    }

    $payload = json_decode($req->rawContent(), true);

    if (!isset($payload['text'])) {
        $res->status(400);
        $res->end(json_encode(['error' => 'Missing field: text']));
        return;
    }

    $text = $payload['text'];

    // Преобразуем в sparse-vector
    $vec = tokensToSparseVector($text, $vocab, $tokenizer);

    // Делаем предикт
    $pred = $nb->predictTopN([$vec], 24);

    $res->header("Content-Type", "application/json");
    $res->header("Access-Control-Allow-Origin", "*");

//    $res->end(json_encode([
//        'category' => $pred[0],
//    ]));

    $res->end(json_encode([
        'top24' => $pred[0],
    ]));
});

$server->start();
