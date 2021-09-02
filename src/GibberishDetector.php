<?php

namespace yamut\GibberishDetector;

use Exception;

/**
 * A simple library to train a cacheable model and detect Gibberish in arbitrary character sets.
 *
 * Note: This library is built using PHP's standard string functions, but should
 * really be refactored to use mb_str libraries for full UTF8 support. This may impact performance.
 *
 * @author  Greg Schoppe <contact@gschoppe.com>
 * @copyright 2019 Licensed for use under the GPL v2.0+
 *
 * @version 2.0 - refactor for proper separation of concern
 */
class GibberishDetector
{
    private const DEFAULT_MODEL = __DIR__ . DIRECTORY_SEPARATOR . 'Model' . DIRECTORY_SEPARATOR . 'gibberish_model.json';
    protected int $minimum_probability = 10;
    protected string $charset = ' abcdefghijklmnopqrstuvwxyz';
    protected array $charmap;
    protected ?array $sequences = null;
    protected float $threshold = 0;

    /**
     * @param string|array|null $model An optional cached model, as exported by `export_model()`
     * @throws Exception
     */
    public function __construct($model = null, bool $useDefault = true)
    {
        if (!empty($model) || $useDefault) {
            if ($useDefault) {
                $model = json_decode(file_get_contents(self::DEFAULT_MODEL), true);
            }
            $this->import_model($model);
        } else {
            $this->build_charmap();
        }
    }

    /**
     * Import a cached model, to bypass the training stage.
     *
     * @param string|array $model A cached model, as exported by `export_model()`. Can be passed as an array for high performance implementations
     *
     * @throws Exception
     */
    public function import_model($model): void
    {
        if (!is_array($model)) {
            $model = unserialize($model);
        }
        if (empty($model) || empty($model['charset']) ||
            empty($model['sequences']) || empty($model['threshold'])) {
            throw new Exception("This is not a valid model.");
        }
        if (strlen($model['charset']) != count($model['sequences'])) {
            throw new Exception("This model's charset does not match the stored sequences.");
        }

        $this->set_charset($model['charset']);
        $this->sequences = $model['sequences'];
        $this->threshold = $model['threshold'];
    }

    /**
     * Export a cached model, for use with future instances of this class.
     *
     * @param bool $serialize If false, return a raw array, rather than a serialized string. Defaults to true.
     *
     * @return string|array An instance of the current model, for use in future instances of this class
     * @throws Exception
     */
    public function export_model(bool $serialize = true)
    {
        if (empty($this->sequences)) {
            throw new Exception("You must train the model before exporting.");
        }
        $model = array(
            'charset' => $this->charset,
            'sequences' => $this->sequences,
            'threshold' => $this->threshold,
        );
        if (!$serialize) {
            return $model;
        }
        return serialize($model);
    }

    /**
     * Load a specific character set for use in training.
     *
     * Note: Uppercase and lowercase characters will be deduplicated before
     * loading the charset. In future versions a flag could be added to toggle
     * this behavior.
     *
     * @param string $charset a string containing all valid characters for training.
     *
     * @throws Exception
     */
    public function set_charset(string $charset): void
    {
        $charset = strtolower($charset);
        $charset = preg_replace('/\s+/', ' ', $charset);
        $charset .= ' '; // ensure that a whitespace character is included
        /** @var string $charset */
        $charset = count_chars($charset, 3); // deduplicate characters
        if (strlen($charset) <= 1) {
            throw new Exception("This character set is invalid.");
        }

        if ($this->charset !== $charset) {
            $this->charset = $charset;

            // invalidate sequences trained from a different charmap
            $this->sequences = null;
            $this->threshold = 0;
        }
        $this->build_charmap();
    }

    /**
     * Test a string to see if it contains gibberish.
     *
     * Note: strings containing characters that do not appear in the configured
     * charset can still be tested for gibberish content, but only the portions
     * of the string containing characters in the charset will be evaluated for
     * gibberish.
     *
     * @param string $candidate_text The string being tested for gibberish
     * @param boolean $verbose If true, this function will return an array containing additional information about the model's determination. defaults to false.
     *
     * @return bool|array Returns a bool if verbose is false, where true signifies a gibberish message. If verbose is true it returns an array, containing additional information about the model's determination.
     * @throws Exception
     */
    public function evaluate(string $candidate_text, bool $verbose = false)
    {
        if (empty($this->sequences)) {
            throw new Exception(
                "No training data has been provided. Load a cached model or train a new model before running evaluate."
            );
        }
        $candidate_text = $this->normalize_string($candidate_text);
        if (empty($candidate_text)) {
            throw new Exception("No candidate text was provided.");
        }
        $probability = $this->evaluate_probability($candidate_text);
        $is_gibberish = $probability <= $this->threshold;

        if ($verbose) {
            return array(
                'is_gibberish' => $is_gibberish,
                'probability' => $probability,
                'threshold' => $this->threshold,
            );
        }
        return $is_gibberish;
    }

    /**
     * Builds the model this class uses to detect gibberish.
     *
     * @param array $dictionary An array of valid non-gibberish text samples, used to train the model. Passed by reference, due to potential size issues.
     * @param array $known_good An array of valid non-gibberish text samples, used to set an expected threshold for success in the model. Passed by reference, due to potential size issues.
     * @param array $known_bad An array of gibberish text samples, used to set an expected threshold for failure in the model. Passed by reference, due to potential size issues.
     * @throws Exception
     */
    public function train(array $dictionary, array $known_good, array $known_bad): void
    {
        $this->build_sequences($dictionary);
        $this->find_threshold($known_good, $known_bad);
    }

    /**
     * @param array $dictionary
     * @throws Exception
     */
    protected function build_sequences(array $dictionary): void
    {
        if (empty($dictionary)) {
            throw new Exception("Training function is missing a necessary dictionary.");
        }

        // build a matrix of character pairs, and seed it with a
        // minimum likelihood of seeing that pair
        $probability_matrix = array();
        $range = range(0, count($this->charmap) - 1);
        foreach ($range as $i) {
            $row = array();
            foreach ($range as $j) {
                $row[$j] = $this->minimum_probability;
            }
            $probability_matrix[$i] = $row;
        }

        // iterate over dictionary building out likelihood of character pairs
        foreach ($dictionary as $line) {
            $line = $this->normalize_string($line);
            $characters = str_split($line);
            $a = false;
            foreach ($characters as $b) {
                if ($a !== false) {
                    $probability_matrix[$this->charmap[$a]][$this->charmap[$b]] += 1;
                }
                $a = $b;
            }
        }

        // convert from number of occurrences to logarithmic probability of occurrence
        foreach ($probability_matrix as $char1 => $row) {
            $total_occurrences = (float)array_sum($row);
            foreach ($row as $char2 => $occurrences) {
                if ($total_occurrences == 0) {
                    $log_value = 0;
                } else {
                    $log_value = log($occurrences / $total_occurrences);
                }
                $probability_matrix[$char1][$char2] = $log_value;
            }
        }

        $this->sequences = $probability_matrix;
    }

    /**
     * @throws Exception
     */
    protected function find_threshold(array $known_good, array $known_bad): void
    {
        if (empty($known_good) || empty($known_bad)) {
            throw new Exception("Training function needs good and bad examples to set threshold.");
        }

        $min_good = PHP_INT_MAX;
        $max_bad = 0;
        foreach ($known_good as $candidate_text) {
            $min_good = min($min_good, $this->evaluate_probability($candidate_text));
        }
        foreach ($known_bad as $candidate_text) {
            $max_bad = max($max_bad, $this->evaluate_probability($candidate_text));
        }

        if ($min_good < $max_bad) {
            throw new Exception("Good content and gibberish are not sufficiently distinguishable.");
        }

        $this->threshold = ($min_good + $max_bad) / 2;
    }

    protected function evaluate_probability(string $candidate_text): float
    {
        $probability_sum = 0;
        $transition_count = 0;
        $characters = str_split($this->normalize_string($candidate_text));

        $a = false;
        foreach ($characters as $b) {
            if ($a !== false) {
                $x = $this->charmap[$a];
                $y = $this->charmap[$b];
                $transition_probability = $this->sequences[$x][$y] ?? 0;
                $probability_sum += $transition_probability;
                $transition_count++;
            }
            $a = $b;
        }

        // The exponentiation translates from log probs to probs.
        return exp($probability_sum / max($transition_count, 1));
    }

    protected function build_charmap(): void
    {
        $this->charmap = array_flip(str_split($this->charset));
    }

    protected function normalize_string(string $str): string
    {
        $str = strtolower($str);
        $pattern = "/[^" . preg_quote($this->charset, "/") . "]/";
        $str = preg_replace($pattern, ' ', $str);
        $str = preg_replace('/\s+/', ' ', $str);
        return trim($str);
    }
}
