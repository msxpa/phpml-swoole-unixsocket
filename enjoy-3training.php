#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/EnjoyNaiveBayes.php';

use Phpml\Tokenization\WhitespaceTokenizer;

// Папки и файлы
$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.php";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";
$vocabSize    = 3000000;
$batchSize    = 10000;
$saveEvery    = 500000;

// Создаём папки
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

if (!file_exists($trainingFile) || filesize($trainingFile) < 10) {
    file_put_contents($trainingFile, "[]");
    echo "⚠️ training_data.json пустой или не найден.\n";
    exit(0);
}

echo "📚 Начинаем обучение из $trainingFile\n";

// ---- 1. Быстрый сбор словаря токенов ----
echo "🔹 Сбор словаря токенов...\n";
$tokenizer = new WhitespaceTokenizer();
$tokenCounts = [];
$totalLines = 0;

$jsonData = json_decode(file_get_contents($trainingFile), true);
if (json_last_error() !== JSON_ERROR_NONE || !is_array($jsonData)) {
    die("❌ Ошибка чтения JSON файла: " . json_last_error_msg() . "\n");
}

$totalLines = count($jsonData);
echo "📊 Найдено $totalLines строк для обработки\n";

// Быстрый подсчет токенов
foreach ($jsonData as $i => $item) {
    if (!isset($item['title'])) continue;
    
    $title = mb_strtolower(trim($item['title']));
    $tokens = $tokenizer->tokenize($title);
    
    foreach ($tokens as $token) {
        $tokenCounts[$token] = ($tokenCounts[$token] ?? 0) + 1;
    }
    
    if (($i + 1) % 50000 == 0) {
        echo "  Токенизировано " . ($i + 1) . " строк...\n";
    }
}

// Оставляем топ-N токенов
arsort($tokenCounts);
$tokenCounts = array_slice($tokenCounts, 0, $vocabSize, true);
$vocab = array_flip(array_keys($tokenCounts));
$vecSize = count($vocab);

echo "✅ Словарь токенов построен, размер: $vecSize\n";

// ---- 2. Оптимизированное обучение батчами ----
echo "🔹 Начало обучения модели...\n";
$nb = new EnjoyNaiveBayes();
$processedLines = 0;

// Функция преобразования текста в sparse vector
function tokensToSparseVector(string $text, array $vocab, WhitespaceTokenizer $tokenizer): array {
    $vec = [];
    $tokens = $tokenizer->tokenize(mb_strtolower(trim($text)));
    foreach ($tokens as $token) {
        if (isset($vocab[$token])) {
            $index = $vocab[$token];
            $vec[$index] = ($vec[$index] ?? 0) + 1;
        }
    }
    return $vec;
}

// Обучаем батчами
$batchVectors = [];
$batchLabels = [];
$batchCount = 0;

foreach ($jsonData as $i => $item) {
    if (!isset($item['title'], $item['category'])) continue;
    
    $vector = tokensToSparseVector($item['title'], $vocab, $tokenizer);
    $category = trim($item['category']);
    
    $batchVectors[] = $vector;
    $batchLabels[] = $category;
    $processedLines++;
    
    // Обучаем батч когда накопилось batchSize строк
    if (count($batchVectors) >= $batchSize) {
        $nb->partialFit($batchVectors, $batchLabels);
        $batchCount++;
        
        echo "  Обучен батч #$batchCount (" . ($batchCount * $batchSize) . " строк)...\n";
        
        // Сбрасываем батч
        $batchVectors = [];
        $batchLabels = [];
        
        // Промежуточное сохранение
        if (($batchCount * $batchSize) % $saveEvery == 0) {
            file_put_contents($modelFile . '.tmp', serialize([
                'vocab' => $vocab,
                'nb'    => $nb
            ]));
            echo "💾 Промежуточная модель сохранена после " . ($batchCount * $batchSize) . " строк\n";
        }
    }
}

// Обучаем последний неполный батч
if (!empty($batchVectors)) {
    $nb->partialFit($batchVectors, $batchLabels);
    $batchCount++;
    echo "  Обучен финальный батч #$batchCount ($processedLines строк)...\n";
}

echo "✅ Обучение завершено, всего обработано $processedLines строк\n";

// ---- 3. Предварительное вычисление вероятностей и сохранение ----
echo "💾 Предварительное вычисление вероятностей...\n";
$nb->precomputeLogProbabilities();

echo "💾 Сохранение финальной модели...\n";
file_put_contents($modelFile, serialize([
    'vocab' => $vocab,
    'nb'    => $nb
]));

// Выводим финальную информацию о модели
$info = $nb->getModelInfo();
echo "📊 Финальная модель:\n";
echo "   - Классы: " . implode(', ', $info['classes']) . "\n";
echo "   - Количество классов: " . count($info['classes']) . "\n";
echo "   - Размер словаря: " . $info['vocabulary_size'] . "\n";
echo "   - Документов по классам:\n";
foreach ($info['class_counts'] as $class => $count) {
    $prior = $info['priors'][$class] * 100;
    echo "     * $class: $count документов (" . number_format($prior, 2) . "%)\n";
}

echo "✅ Модель сохранена: $modelFile\n";

// ---- 4. Архивируем данные ----
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
file_put_contents($usedFile, json_encode($jsonData, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
file_put_contents($trainingFile, "[]");

echo "📦 Данные перемещены в: $usedFile\n";
echo "🧹 training_data.json очищен\n";

// Удаляем временный файл если существует
if (file_exists($modelFile . '.tmp')) {
    unlink($modelFile . '.tmp');
}

$totalTime = round(microtime(true) - $_SERVER["REQUEST_TIME_FLOAT"], 2);
echo "✨ Обучение завершено успешно! Обработано $processedLines строк за $totalTime сек.\n";
