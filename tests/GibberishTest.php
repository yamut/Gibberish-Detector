<?php

namespace yamut\GibberishDetector\tests;

use PHPUnit\Framework\TestCase;
use yamut\GibberishDetector\GibberishDetector;

class GibberishTest extends TestCase
{
    public function testTrain()
    {
        $gibberishDetector = new GibberishDetector();
        $gibberishDetector->train(
            file('src/TrainingData/dictionary.txt'),
            file('src/TrainingData/good.txt'),
            file('src/TrainingData/bad.txt'),
        );
        $model = $gibberishDetector->export_model(false);
        $this->assertIsArray($model);
        foreach (['charset', 'sequences', 'threshold'] as $key) {
            $this->assertArrayHasKey($key, $model);
        }
        $gibReflection = new \ReflectionClass($gibberishDetector);
        $charsetProperty = $gibReflection->getProperty('charset');
        $charsetProperty->setAccessible(true);
        $charset = $charsetProperty->getValue($gibberishDetector);
        $this->assertCount(strlen($charset), $model['sequences']);
    }

    public function providesPredictionTestData()
    {
        return [
            ['john smith', false],
            ['kjdjksdf', true],
        ];
    }

    /**
     * @param string $term
     * @param bool $result
     * @throws \Exception
     * @dataProvider providesPredictionTestData
     */
    public function testPrediction(string $term, bool $result)
    {
        $gibberishDetector = new GibberishDetector();
        $gibberishDetector->train(
            file('src/TrainingData/dictionary.txt'),
            file('src/TrainingData/good.txt'),
            file('src/TrainingData/bad.txt'),
        );
        $model = $gibberishDetector->export_model(true);
        $gibberishDetector = new GibberishDetector($model);
        $this->assertEquals($result, $gibberishDetector->evaluate($term));
    }
}