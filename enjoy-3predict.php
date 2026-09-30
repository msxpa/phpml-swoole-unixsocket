#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/EnjoyNaiveBayes.php';

use Phpml\Tokenization\WhitespaceTokenizer;

$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.php";

if (!file_exists($modelFile)) {
    die("❌ Модель не найдена. Сначала запусти trainer.\n");
}

// Замер времени загрузки модели
$startLoad = microtime(true);
$modelData = unserialize(file_get_contents($modelFile));
$vocab = $modelData['vocab'];
/** @var EnjoyNaiveBayes $nb */
$nb = $modelData['nb'];
$loadTime = microtime(true) - $startLoad;

$tokenizer = new WhitespaceTokenizer();

if ($argc < 2) {
    die("Использование: php enjoy-predict.php \"текст\"\n");
}

$query = mb_strtolower(trim($argv[1]));

// Transform text to sparse vector
function textToSparseVector(string $text, array $vocab, WhitespaceTokenizer $tokenizer): array {
    $vec = [];
    foreach ($tokenizer->tokenize($text) as $token) {
        if (isset($vocab[$token])) {
            $index = $vocab[$token];
            $vec[$index] = ($vec[$index] ?? 0) + 1;
        }
    }
    return $vec;
}

$startVector = microtime(true);
$vector = textToSparseVector($query, $vocab, $tokenizer);
$vectorTime = microtime(true) - $startVector;

$startPredict = microtime(true);
$top3 = $nb->predictTopN([$vector], 10);
$predictTime = microtime(true) - $startPredict;

$totalTime = microtime(true) - $startLoad;

echo "🔍 Текст: $query\n";
echo "⚡ Производительность:\n";
echo "   - Загрузка модели: " . round($loadTime * 1000, 2) . "мс\n";
echo "   - Векторизация: " . round($vectorTime * 1000, 2) . "мс\n";
echo "   - Предсказание: " . round($predictTime * 1000, 2) . "мс\n";
echo "   - Общее время: " . round($totalTime * 1000, 2) . "мс\n";
echo "📊 Топ-3 категорий:\n";

foreach ($top3[0] as $cat => $prob) {
    printf("  - %s: %.4f\n", $cat, $prob);
}
