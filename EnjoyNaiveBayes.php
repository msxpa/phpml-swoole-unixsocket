<?php

class EnjoyNaiveBayes
{
    private array $classes = [];
    private array $theta = [];
    private array $priors = [];
    private array $classCounts = [];
    private array $totalTokensPerClass = [];
    private array $logProbabilities = []; // КЭШ логарифмов
    private int $vocabSize = 0;
    private bool $isPrecomputed = false;

    public function partialFit(array $samples, array $labels)
    {
        foreach ($labels as $i => $c) {
            $sample = $samples[$i];

            if (!isset($this->classes[$c])) {
                $this->classes[$c] = $c;
                $this->theta[$c] = [];
                $this->priors[$c] = 0.0;
                $this->classCounts[$c] = 0;
                $this->totalTokensPerClass[$c] = 0;
            }

            foreach ($sample as $tokenIndex => $count) {
                $this->theta[$c][$tokenIndex] = ($this->theta[$c][$tokenIndex] ?? 0) + $count;
                $this->totalTokensPerClass[$c] += $count;
                
                // Обновляем vocabSize
                if ($tokenIndex >= $this->vocabSize) {
                    $this->vocabSize = $tokenIndex + 1;
                }
            }

            $this->classCounts[$c]++;
        }

        $totalDocs = array_sum($this->classCounts);
        foreach ($this->classes as $c) {
            $this->priors[$c] = $this->classCounts[$c] / $totalDocs;
        }
        
        $this->isPrecomputed = false; // Сбрасываем кэш
    }

    // Предварительное вычисление логарифмов (вызывать после обучения)
    public function precomputeLogProbabilities(): void
    {
        $this->logProbabilities = [];
        
        foreach ($this->classes as $c) {
            $totalCount = $this->totalTokensPerClass[$c] + $this->vocabSize;
            $this->logProbabilities[$c] = [];
            
            // Вычисляем логарифмы для всех известных токенов
            foreach ($this->theta[$c] as $tokenIndex => $count) {
                $probability = ($count + 1) / $totalCount;
                $this->logProbabilities[$c][$tokenIndex] = log($probability);
            }
            
            // Логарифм для неизвестных токенов (вычисляем один раз)
            $this->logProbabilities[$c]['_unknown'] = log(1 / $totalCount);
        }
        
        $this->isPrecomputed = true;
    }

    public function predictTopN(array $samples, int $N = 3): array
    {
        if (!$this->isPrecomputed) {
            $this->precomputeLogProbabilities();
        }

        $results = [];

        foreach ($samples as $sample) {
            $scores = [];

            foreach ($this->classes as $c) {
                $log_prior = log($this->priors[$c] ?? 1e-12);
                $log_likelihood = 0.0;

                foreach ($sample as $tokenIndex => $count) {
                    if (isset($this->logProbabilities[$c][$tokenIndex])) {
                        $log_likelihood += $count * $this->logProbabilities[$c][$tokenIndex];
                    } else {
                        // Используем предвычисленное значение для неизвестных токенов
                        $log_likelihood += $count * $this->logProbabilities[$c]['_unknown'];
                    }
                }

                $scores[$c] = $log_prior + $log_likelihood;
            }

            // Softmax
            $maxLog = max($scores);
            $expScores = array_map(fn($v) => exp($v - $maxLog), $scores);
            $sumExp = array_sum($expScores);
            $probs = array_map(fn($v) => $v / $sumExp, $expScores);

            arsort($probs);
            $results[] = array_slice($probs, 0, $N, true);
        }

        return $results;
    }

    private function getVocabularySize(): int
    {
        return $this->vocabSize;
    }

    public function getModelInfo(): array
    {
        return [
            'classes' => array_keys($this->classes),
            'class_counts' => $this->classCounts,
            'vocabulary_size' => $this->vocabSize,
            'priors' => $this->priors
        ];
    }
    
    public function __sleep() {
        return ['classes', 'theta', 'priors', 'classCounts', 'totalTokensPerClass', 'vocabSize'];
    }
    
    public function __wakeup() {
        $this->precomputeLogProbabilities();
    }
}
