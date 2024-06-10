import cv2
import numpy as np
import os
import json

LINIJE = [
    (0, 2, (0, 0, 255)),
    (0, 5, (0, 0, 255)),
    (5, 8, (0, 0, 255)),
    (2, 7, (0, 0, 225)),
    (9, 10, (0, 0, 225)),
    (11, 12, (0, 0, 225)),
    (11, 13, (0, 0, 225)),
    (11, 23, (0, 0, 225)),
    (13, 15, (0, 0, 225)),
    (15, 21, (0, 0, 225)),
    (15, 17, (0, 0, 225)),
    (15, 19, (0, 0, 225)),
    (17, 19, (0, 0, 225)),
    (12, 14, (0, 0, 225)),
    (12, 24, (0, 0, 225)),
    (15, 19, (0, 0, 225)),
    (14, 16, (0, 0, 225)),
    (16, 22, (0, 0, 225)),
    (16, 18, (0, 0, 225)),
    (18, 20, (0, 0, 225)),
    (23, 24, (0, 0, 225)),
    (23, 25, (0, 0, 225)),
    (25, 27, (0, 0, 225)),
    (27, 29, (0, 0, 225)),
    (27, 31, (0, 0, 225)),
    (29, 31, (0, 0, 225)),

    (24, 26, (0, 0, 225)),
    (26, 28, (0, 0, 225)),
    (28, 30, (0, 0, 225)),
    (28, 32, (0, 0, 225)),
    (30, 32, (0, 0, 225)),
   
]


def iscrtaj_markere(img, markeri, threshold=None, radius=5, thickness=3, color=(255, 0, 0)):
    p = markeri.shape[0]
    result = np.copy(img)
    for m in range(p):
        if threshold == None or markeri[m, 2] > threshold:
            cv2.circle(result, (int(markeri[m, 0]), int(
                markeri[m, 1])), radius, color, thickness=thickness)
    for l in LINIJE:
        if threshold == None or ((markeri[l[0], 2] > threshold) and (markeri[l[1], 2] > threshold)):
            cv2.line(result, (int(markeri[l[0], 0]), int(markeri[l[0], 1])), (int(
                markeri[l[1], 0]), int(markeri[l[1], 1])), color=l[2], thickness=thickness)
    return result


def ucitaj_fajlove(folder, sort=True, ekstenzije=None):
    all_files = list()
    for f in os.listdir(folder):
        if ekstenzije == None or (os.path.splitext(f)[-1] in ekstenzije):
            all_files.append(os.path.join(folder, f))
    if sort:
        all_files.sort()
    return all_files


def ucitaj_unutrasnje_parametre(calib_file):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeffs = fs.getNode('distortion').mat()
    fs.release()
    return (cam_mat, dist_coeffs)


def ucitaj_sve_parametre(calib_file):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeffs = fs.getNode('distortion').mat()
    Rot = fs.getNode('Rot').mat()
    trans = fs.getNode('Trans').mat()
    fs.release()
    return (cam_mat, dist_coeffs, Rot, trans)


def zarotiraj_markere(markeri, ugao, sirina, visina):
    rezultat = markeri.copy()
    if (ugao == 0):
        pass
    elif (ugao == 90):
        rezultat[:, 0] = sirina - markeri[:, 1]
        rezultat[:, 1] = markeri[:, 0]
    elif (ugao == 180):
        rezultat[:, 0] = sirina - markeri[:, 0]
        rezultat[:, 1] = visina - markeri[:, 1]
    elif (ugao == 270):
        rezultat[:, 0] = markeri[:, 1]
        rezultat[:, 1] = sirina - markeri[:, 0]
    else:
        raise Exception(
            "Uglovi za koje može da se zarotira su 0, 90, 180 i 270")
    return rezultat


def ucitaj_json(f, key_name, rows=3):
    try:
        with open(f) as json_file:
            data = json.load(json_file)
            keypoints = np.matrix(
                data["people"][0][key_name]).reshape((-1, rows))
            return keypoints
    except:
        raise Exception("Nisam mogao da ucitam markere sa kljucem {key_name} iz fajla {filename}".format(
            key_name=key_name, filename=f))
        


def ucitaj_markere(f, rotacija, width, height):
    keypoints = ucitaj_json(f, "pose_keypoints_2d")
    keypoints = zarotiraj_markere(keypoints, rotacija, width, height)
    return keypoints


def projektuj_tacke(tacke, camera_matrix, rot_mat, translation):
    image_points = np.matmul(
        camera_matrix, np.matmul(rot_mat, tacke)+translation)
    return (image_points/image_points[2, :])[0:2, :]


def ucitaj_projektuj_3d_markere(m3d_file, cam_mat, rot, trans):
    keypoints = ucitaj_json(m3d_file, "pose_keypoints_3d", rows=4)
    tacke = np.zeros((keypoints.shape[0], 3), dtype=float)
    tacke[:, 0:2] = projektuj_tacke(keypoints[:, 0:3].T, cam_mat, rot, trans).T
    tacke[:, 2] = keypoints[:, -1].T
    return tacke

def sacuvaj_pozu(pose, i, filename_template):
    name = filename_template.format(num=i)
    poza = {"people": [{"pose_keypoints_3d": pose.ravel().tolist()}]}
    json_object = json.dumps(poza, indent=2)

    if not os.path.isdir(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))

    with open(name, "w") as outfile:
        outfile.write(json_object)