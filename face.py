import cv2
import face_recognition

# Load the known faces and their names
known_faces = [
    face_recognition.load_image_file("person1.jpg"),
    face_recognition.load_image_file("person2.jpg"),
    face_recognition.load_image_file("person3.jpg"),
]
known_names = ["Person 1", "Person 2", "Person 3"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Load the image we want to check
unknown_image = face_recognition.load_image_file("unknown.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Loop through each face found in the unknown image
for face_encoding in face_encodings:
    # See if the face is a match for the known faces
    matches = face_recognition.compare_faces(known_faces, face_encoding)
    name = "Unknown"

    # If a match was found in known_faces, use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]

    face_names.append(name)

# Draw rectangles around the faces
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(unknown_image, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

# Show the final image with the face recognition results
cv2.imshow("Face Recognition Results", unknown_image)
cv2.waitKey(0)
