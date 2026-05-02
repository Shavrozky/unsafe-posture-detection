import numpy as np


class PostureRules:
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def classify(self, keypoints):
        """
        keypoints shape: (17, 2)
        return: posture label string
        """

        if keypoints is None or len(keypoints) < 17:
            return "unknown"

        shoulder = self._midpoint(
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.RIGHT_SHOULDER]
        )
        hip = self._midpoint(
            keypoints[self.LEFT_HIP],
            keypoints[self.RIGHT_HIP]
        )
        knee = self._midpoint(
            keypoints[self.LEFT_KNEE],
            keypoints[self.RIGHT_KNEE]
        )
        ankle = self._midpoint(
            keypoints[self.LEFT_ANKLE],
            keypoints[self.RIGHT_ANKLE]
        )

        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]

        if self._is_invalid(shoulder) or self._is_invalid(hip):
            return "unknown"

        body_dx = abs(shoulder[0] - hip[0])
        body_dy = abs(shoulder[1] - hip[1])

        # 1. Hands up / emergency gesture
        if self._hands_up(left_wrist, right_wrist, shoulder, hip):
            return "hands_up"

        # 2. Lying / fall
        # Pada image coordinate, y makin besar berarti makin ke bawah.
        # Jika badan lebih horizontal dibanding vertical, kemungkinan lying/fall.
        if body_dx > body_dy * 1.4:
            return "lying_fall"

        # 3. Bending
        # Shoulder jauh lebih turun mendekati hip/knee atau torso sangat condong.
        if body_dx > body_dy * 0.55:
            return "bending"

        # 4. Sitting
        if not self._is_invalid(knee) and not self._is_invalid(ankle):
            hip_to_knee = abs(hip[1] - knee[1])
            knee_to_ankle = abs(knee[1] - ankle[1])

            # Saat duduk, hip cenderung mendekati knee secara vertikal.
            if hip_to_knee < knee_to_ankle * 0.75:
                return "sitting"

        # 5. Default
        return "standing"

    def is_unsafe(self, label):
        return label in ["lying_fall", "bending", "hands_up"]

    def _hands_up(self, left_wrist, right_wrist, shoulder, hip):
        """
        Minimal satu tangan terangkat di atas shoulder dengan margin 10% dari
        tinggi torso. Margin kecil ini mencegah false positive dari noise
        keypoint tanpa mengorbankan sensitivitas gesture darurat.
        """
        torso_height = abs(shoulder[1] - hip[1])
        margin = torso_height * 0.10

        left_up = (
            not self._is_invalid(left_wrist)
            and left_wrist[1] < shoulder[1] - margin
        )
        right_up = (
            not self._is_invalid(right_wrist)
            and right_wrist[1] < shoulder[1] - margin
        )
        # Cukup salah satu tangan terangkat untuk trigger (emergency gesture).
        return left_up or right_up

    def _midpoint(self, p1, p2):
        if self._is_invalid(p1):
            return p2
        if self._is_invalid(p2):
            return p1
        return (p1 + p2) / 2.0

    def _is_invalid(self, point):
        if point is None:
            return True
        return np.allclose(point, 0)