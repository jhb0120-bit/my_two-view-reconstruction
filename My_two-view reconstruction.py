import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 카메라 파라미터
fx = fy = 1086.0
cx, cy = 512.0, 384.0
k1 = -0.0568965  # Brown-Conrady radial distortion 1-term
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)
dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)

# Point 개수
MAX_POINTS = 40


# 특징점 추출 + 매칭
def detect_and_match(img1_gray, img2_gray):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # FLANN 기반 매칭[web:40][web:43]
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []

    # Lowe ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    return pts1, pts2, good_matches, kp1, kp2

def draw_matches(img1, img2, kp1, kp2, matches, max_draw=50):
    draw = matches
    if len(matches) > max_draw:
        draw = matches[:max_draw]
    matched_vis = cv.drawMatches(img1, kp1, img2, kp2, draw, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("matches", matched_vis)
    cv.waitKey(1)
    return matched_vis


# 왜곡 보정
def undistort_points(pts, K, dist_coeffs):
    pts_undist = cv.undistortPoints(pts.reshape(-1, 1, 2), K, dist_coeffs)
    return pts_undist.reshape(-1, 2)


# Essential, Pose, Triangulation
def estimate_essential_and_pose(pts1, pts2, K):
    # RANSAC 기반 Essential 추정[web:3][web:46]
    E, mask = cv.findEssentialMat(pts1, pts2, K,
                                  method=cv.RANSAC,
                                  prob=0.999,
                                  threshold=1.0)
    # recoverPose로 R, t 계산[web:3][web:24]
    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
    return E, R, t, mask_pose

def triangulate_points(pts1, pts2, K, R, t):
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    pts1_h = pts1.T
    pts2_h = pts2.T
    X_h = cv.triangulatePoints(P0, P1, pts1_h, pts2_h)
    X = (X_h[:3, :] / X_h[3, :]).T
    return X


# 3D 시각화 (Matplotlib)
def visualize_3d_points(X, R=None, t=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, c='b', marker='o')

    cam0 = np.array([[0, 0, 0]])
    ax.scatter(cam0[:, 0], cam0[:, 1], cam0[:, 2], c='r', s=50)
    ax.text(0, 0, 0, "Cam0", color='r')

    if R is not None and t is not None:
        cam1_center = -R.T @ t
        cam1_center = cam1_center.ravel()
        ax.scatter(cam1_center[0], cam1_center[1], cam1_center[2], c='g', s=50)
        ax.text(cam1_center[0], cam1_center[1], cam1_center[2], "Cam1", color='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.show()


def main():
    # 이미지 로드
    img1 = cv.imread("D:/OpenCV/opencv/reconstruction_data/003.jpg")
    img2 = cv.imread("D:/OpenCV/opencv/reconstruction_data/005.jpg")
    if img1 is None or img2 is None:
        print("이미지를 찾을 수 없습니다.")
        return

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 자동 특징 추출 및 매칭
    pts1_raw, pts2_raw, matches, kp1, kp2 = detect_and_match(img1_gray, img2_gray)
    print("전체 매칭 개수:", len(matches))

    # 사용할 점 개수 제한
    if len(pts1_raw) > MAX_POINTS:
        pts1_raw = pts1_raw[:MAX_POINTS]
        pts2_raw = pts2_raw[:MAX_POINTS]
        matches = matches[:MAX_POINTS]
    print("실제 사용 점 개수:", len(pts1_raw))

    matched_vis = draw_matches(img1, img2, kp1, kp2, matches)
    cv.imwrite("vis_matches.png", matched_vis)

    # 왜곡 보정 좌표
    pts1_undist = undistort_points(pts1_raw, K, dist_coeffs)
    pts2_undist = undistort_points(pts2_raw, K, dist_coeffs)

    # Essential 행렬 & Pose
    # recoverPose는 픽셀 좌표 + K를 인자로 받으므로 pts*_raw 사용[web:3][web:24]
    E, R, t, mask_pose = estimate_essential_and_pose(pts1_raw, pts2_raw, K)

    print("Estimated Essential matrix E:")
    print(E)
    print("Rotation R:")
    print(R)
    print("Translation t (up to scale):")
    print(t)

    np.savetxt("E_matrix.txt", E)
    np.savetxt("R_matrix.txt", R)
    np.savetxt("t_vector.txt", t)

    # 인라이어만 사용해서 삼각측량
    mask_pose = mask_pose.ravel().astype(bool)
    pts1_in = pts1_raw[mask_pose]
    pts2_in = pts2_raw[mask_pose]

    if len(pts1_in) < 5:
        print("MAX_POINTS를 늘려보세요.")
        return

    X = triangulate_points(pts1_in, pts2_in, K, R, t)
    np.savetxt("points3d.txt", X)

    print("재구성된 3D 점 개수:", X.shape[0])

    # 3D 시각화 (여러 viewpoint는 azim/elev만 바꿔서 캡처)
    visualize_3d_points(X, R, t)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
