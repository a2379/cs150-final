import random


class RhythmGenerator:
    """
    A class to handle the rhythm generation for the music composition project,
    using emotion-based stochastic binary subdivision.
    """

    def __init__(self, subdivision_probability=0.3):
        """
        Constructor to initialize the rhythm generator.

        Parameters:
        -----------
        subdivision_probability : float
            The probability of subdividing a note at each level.
        """
        self.subdivision_probability = subdivision_probability

        # Dictionary to store measures with their emotional scores
        self.measure_scores = {}

        self.all_measures = []

    def apply_stochastic_subdivision(self, input_measure, max_depth=2):
        """
        Apply stochastic binary subdivision to a measure.

        Parameters:
        -----------
        input_measure: list
            A list of notes or chords (strings or tuples) to be rhythmically processed.
            Example: ['C4', 'D4', 'E4'] or [('C4', 'E4', 'G4'), 'D4', 'B4']

        max_depth : int
            Maximum subdivision depth. For example:
            - Starting with quarter notes:
              depth=1: eighth notes
              depth=2: sixteenth notes

        Returns:
        --------
        list
            A list of tuples (note, duration) where duration is relative to the smallest
            subdivision. For example, with max_depth=2, a quarter note would have
            duration=8 (4 sixteenth notes).
        """
        # Start with all notes at the longest duration
        initial_duration = 2**max_depth
        rhythm_pattern = (
            [(note, initial_duration) for note in input_measure]
            if input_measure
            else [((), initial_duration)]
        )

        # Process the pattern, subdividing notes recursively
        final_pattern = []
        for note, duration in rhythm_pattern:
            final_pattern.extend(self._subdivide_note(note, duration, 0, max_depth))

        return final_pattern

    def _subdivide_note(self, note, duration, current_depth, max_depth):
        """
        Recursively subdivide a note based on probability.

        Parameters:
        -----------
        note : str or tuple
            The note or chord to potentially subdivide.
        duration : int
            The current duration of the note (in terms of smallest subdivision unit).
        current_depth : int
            The current recursion depth.
        max_depth : int
            The maximum allowed subdivision depth.

        Returns:
        --------
        list
            A list of (note, duration) tuples after subdivision.
        """
        # Base case: reached maximum depth or cannot divide further
        if current_depth >= max_depth or duration <= 1:
            return [(note, duration)]

        # Recursive case
        if random.random() < self.subdivision_probability:
            # Binary subdivision: split into two equal parts
            half_duration = duration // 2
            first_half = self._subdivide_note(
                note, half_duration, current_depth + 1, max_depth
            )
            second_half = self._subdivide_note(
                note, half_duration, current_depth + 1, max_depth
            )

            return first_half + second_half
        else:
            # No subdivision
            return [(note, duration)]

    def calculate_emotional_score(
        self, processed_measure, measure_idx=None, time_signature=(4, 4)
    ):
        """
        Calculate the emotional score of a rhythmic measure based on multiple dimensions.

        Parameters:
        -----------
        processed_measure : list
            A list of tuples (note, duration) representing a processed measure.
        measure_idx : int, optional
            The index of the current measure in the sequence.
        time_signature : tuple, optional
            The time signature of the measure as (numerator, denominator).

        Returns:
        --------
        float
            The total emotional score of the measure.
        """
        # Convert processed_measure to a binary pattern for easier analysis
        pattern = self._convert_to_binary_pattern(processed_measure)

        # Calculate individual scores
        density_score = self._calculate_density_score(pattern)
        syncopation_score = self._calculate_syncopation_score(pattern, time_signature)

        # Calculate surprise score if we have a previous measure
        surprise_score = 0
        if measure_idx is not None and measure_idx > 0:
            prev_pattern = self._convert_to_binary_pattern(
                self.all_measures[measure_idx - 1]
            )
            surprise_score = self._calculate_surprise_score(pattern, prev_pattern)

        # Calculate phrase position multiplier
        phrase_pos_multiplier = 1.0
        if measure_idx is not None:
            phrase_pos_multiplier = self._calculate_phrase_position_multiplier(
                measure_idx
            )

        # Calculate rhythmic development score
        # development_score = self._calculate_development_score(pattern, measure_idx)

        # Combine scores
        base_score = (density_score + syncopation_score + surprise_score) / 3
        total_score = base_score * phrase_pos_multiplier

        return total_score

    def _convert_to_binary_pattern(self, processed_measure):
        """
        Convert a processed measure to a binary pattern (1 for note onset, 0 for rest/continuation).

        Parameters:
        -----------
        processed_measure : list
            A list of tuples (note, duration) representing a processed measure.

        Returns:
        --------
        list
            A binary list where 1 indicates a note onset and 0 indicates a rest or note continuation.
        """
        # Determine the total length of the pattern based on the smallest subdivision
        total_length = sum(duration for _, duration in processed_measure)
        binary_pattern = [0] * total_length

        # Mark note onsets
        position = 0
        for note, duration in processed_measure:
            if note is not None:
                binary_pattern[position] = 1
            position += duration

        return binary_pattern

    def _calculate_density_score(self, pattern):
        """
        Calculate the density score (0-10) based on the ratio of notes to total positions.

        Parameters:
        -----------
        pattern : list
            A binary pattern where 1 indicates a note onset and 0 indicates a rest or continuation.

        Returns:
        --------
        float
            The density score from 0 to 10.
        """
        if not pattern:
            return 0

        density = sum(pattern) / len(pattern)
        return density * 10

    def _calculate_syncopation_score(self, pattern, time_signature=(4, 4)):
        """
        Calculate the syncopation score (0-10) based on emphasis of off-beats.

        Parameters:
        -----------
        pattern : list
            A binary pattern where 1 indicates a note onset and 0 indicates a rest or continuation.
        time_signature : tuple
            The time signature as (numerator, denominator).

        Returns:
        --------
        float
            The syncopation score from 0 to 10.
        """
        if not pattern or sum(pattern) == 0:
            return 0

        # Assumes 4/4 time with 16 sixteenth notes
        # Higher weights for traditionally weaker beats
        if time_signature == (4, 4) and len(pattern) == 4:
            position_weights = [1, 4, 2, 3]
        else:
            # For other time signatures or subdivisions, create a generic weighting
            # where strong beats get weight 1 and all others get higher weights
            positions_per_beat = len(pattern) // time_signature[0]
            position_weights = []
            for beat in range(time_signature[0]):
                for pos in range(positions_per_beat):
                    if pos == 0:  # Strong beat
                        position_weights.append(1)
                    else:  # Off-beat, gradually increase weight for later positions within each beat
                        position_weights.append(1 + (3 * pos / positions_per_beat))

        # Calculate weighted syncopation
        weighted_sum = sum(
            weight * note for weight, note in zip(position_weights, pattern)
        )

        # Only count positions with notes
        note_positions = [i for i, note in enumerate(pattern) if note == 1]
        if not note_positions:
            return 0

        weights_of_note_positions = sum(position_weights[i] for i in note_positions)

        # Normalize to 0-10 scale
        syncopation_score = (
            weighted_sum / weights_of_note_positions
        ) * 2.5  # Scale factor to get near 0-10
        return min(10, syncopation_score)  # Cap at 10

    def _calculate_surprise_score(self, current_pattern, previous_pattern):
        """
        Calculate the rhythmic surprise score (0-10) compared to the previous pattern.
        Handles variable-length patterns through pattern statistics.

        Parameters:
        -----------
        current_pattern : list
            The binary pattern of the current measure.
        previous_pattern : list
            The binary pattern of the previous measure.

        Returns:
        --------
        float
            The surprise score from 0 to 10.
        """
        if not current_pattern or not previous_pattern:
            return 0

        # Compare rhythmic density (percentage of positions with notes)
        current_density = sum(current_pattern) / len(current_pattern)
        previous_density = sum(previous_pattern) / len(previous_pattern)
        density_change = abs(current_density - previous_density)

        # Compare rhythmic grouping (average distance between note onsets)
        current_onsets = [i for i, note in enumerate(current_pattern) if note == 1]
        previous_onsets = [i for i, note in enumerate(previous_pattern) if note == 1]

        # Calculate average distances between consecutive onsets
        current_distances = []
        previous_distances = []

        if len(current_onsets) > 1:
            current_distances = [
                current_onsets[i + 1] - current_onsets[i]
                for i in range(len(current_onsets) - 1)
            ]

        if len(previous_onsets) > 1:
            previous_distances = [
                previous_onsets[i + 1] - previous_onsets[i]
                for i in range(len(previous_onsets) - 1)
            ]

        # Calculate average distance
        current_avg_distance = (
            sum(current_distances) / len(current_distances) if current_distances else 0
        )
        previous_avg_distance = (
            sum(previous_distances) / len(previous_distances)
            if previous_distances
            else 0
        )

        # Normalize the average distance to the pattern length
        if current_avg_distance and previous_avg_distance:
            current_norm_distance = current_avg_distance / len(current_pattern)
            previous_norm_distance = previous_avg_distance / len(previous_pattern)
            distance_change = abs(current_norm_distance - previous_norm_distance)
        else:
            distance_change = 0.5  # Default medium change if we can't calculate

        # Calculate position of first onset relative to pattern length
        current_first_pos = (
            current_onsets[0] / len(current_pattern) if current_onsets else 0
        )
        previous_first_pos = (
            previous_onsets[0] / len(previous_pattern) if previous_onsets else 0
        )
        first_pos_change = abs(current_first_pos - previous_first_pos)

        # Combine metrics to get overall surprise score
        surprise_score = (
            density_change * 3 + distance_change * 4 + first_pos_change * 3
        ) * 10

        return min(10, surprise_score)  # Cap at 10

    def _calculate_phrase_position_multiplier(self, measure_idx, phrase_length=8):
        """
        Calculate a multiplier based on position within a musical phrase.

        Parameters:
        -----------
        measure_idx : int
            The index of the current measure.
        phrase_length : int
            The length of a typical phrase in measures.

        Returns:
        --------
        float
            A multiplier between 0.8 and 1.5.
        """
        phrase_position = measure_idx % phrase_length

        # Define multipliers for different positions in the phrase
        if phrase_position == 1:  # Early in phrase (measures 1)
            return 0.8
        elif phrase_position == 2:  # Building (2)
            return 1.2
        elif phrase_position == 3:  # Climax (measure 3)
            return 1.5
        else:  # Resolution (measure 4)
            return 1.0

    def get_top_emotional_measures(self, count=8):
        """
        Get the indices of the top scoring measures.

        Parameters:
        -----------
        count : int
            The number of top measures to return.

        Returns:
        --------
        list
            A list of indices for the top-scoring measures.
        """
        if not self.measure_scores:
            return []

        # Sort measures by score in descending order
        sorted_measures = sorted(
            self.measure_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return the indices of the top measures
        return [idx for idx, _ in sorted_measures[:count]]

    def process_measure(
        self, measure, max_depth=2, measure_idx=None, time_signature=(4, 4)
    ):
        """
        Process a single measure using stochastic binary subdivision and calculate its emotional score.

        Parameters:
        -----------
        measure : list
            A list of notes or chords to process.
        max_depth : int
            Maximum subdivision depth.
        measure_idx : int, optional
            The index of the current measure in the sequence.
        time_signature : tuple, optional
            The time signature of the measure as (numerator, denominator).

        Returns:
        --------
        list
            A list of tuples (note, duration) after subdivision.
        """
        processed_measure = self.apply_stochastic_subdivision(measure, max_depth)

        # Store this measure for future reference
        if measure_idx is None:
            measure_idx = len(self.all_measures)

        self.all_measures.append(processed_measure)

        # Calculate and store emotional score
        emotional_score = self.calculate_emotional_score(
            processed_measure, measure_idx, time_signature
        )
        self.measure_scores[measure_idx] = emotional_score

        return processed_measure

    def arrange_piece(
        self, input_measures, total_measures=16, max_depth=2, time_signature=(4, 4)
    ):
        """
        Arrange a complete piece by processing input measures and organizing them based on emotional scores.

        Parameters:
        -----------
        input_measures : list
            A list of lists, where each inner list contains notes/chords for a measure.
        total_measures : int
            The total number of measures in the final piece.
        max_depth : int
            Maximum subdivision depth for stochastic binary subdivision.
        time_signature : tuple
            The time signature as (numerator, denominator).

        Returns:
        --------
        list
            A list of processed measures arranged according to emotional scoring rules.
        """
        self.all_measures = []
        self.measure_scores = {}

        # Process all input measures to get their emotional scores
        measure_list = [
            input_measures[i : i + 4] for i in range(0, len(input_measures), 4)
        ]
        for i, measure in enumerate(measure_list):
            self.process_measure(measure, max_depth, i, time_signature)

        final_arrangement = []
        num_phrases = (total_measures - 4) // 4

        # Process each regular phrase (all but the last one)
        for phrase in range(num_phrases):
            for position in range(4):
                measure_idx = phrase * 4 + position

                if (
                    position == 3
                ):  # At each 4th measure, use highest emotional rhythm so far
                    candidate_indices = range(len(self.all_measures))
                    sorted_indices = sorted(
                        candidate_indices,
                        key=lambda idx: self.measure_scores.get(idx, 0),
                        reverse=True,
                    )
                    selected_idx = sorted_indices[0]
                    final_arrangement.append(self.all_measures[selected_idx])
                else:
                    # Use regular measure
                    final_arrangement.append(self.all_measures[measure_idx])

        # For the last 4 measures, use the 4 highest emotional rhythms in ascending order
        top_indices = self.get_top_emotional_measures(4)
        top_indices_ascending = sorted(
            top_indices, key=lambda idx: self.measure_scores.get(idx, 0)
        )

        for idx in top_indices_ascending:
            final_arrangement.append(self.all_measures[idx])

        return final_arrangement
