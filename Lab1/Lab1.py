
from pickle import NONE
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
# Rotate object
def rotate(points, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    result=  np.dot(points, rotation_matrix.T)
    print("Result:")
    print(result)
    return result
def rotate_cv(points, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)
    result = cv.transform(points.astype(np.float64).reshape(-1, 1, 2), rotation_matrix.reshape(2, 2)).reshape(-1, 2)
    print("Result:")
    print(result)
    return result

#Scale object 

def scale(points, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])
    result= np.dot(points, scaling_matrix)
    print("Result:")
    print(result)
    return result
def scale_cv(points, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])
    result = cv.transform(points.astype(np.float64).reshape(-1, 1, 2), scaling_matrix.reshape(2, 2)).reshape(-1, 2)
    print("Result:")
    print(result)
    return result
#reflect object
def reflect(points, axis):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x' or 'y'")
    result= np.dot(points, reflection_matrix)
    print("Result:")
    print(result)
    return result
def reflect_cv(points, axis):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x' or 'y'")
    result = cv.transform(points.astype(np.float64).reshape(-1, 1, 2), reflection_matrix.reshape(2, 2)).reshape(-1, 2)
    print("Result:")
    print(result)
    return result
# shear object
def shear(points, axis, shear_factor):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear_factor],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [shear_factor, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x' or 'y'")
    result =np.dot(points, shear_matrix)
    print("Result:")
    print(result)
    return result
def shear_cv(points, axis, shear_factor):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear_factor],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [shear_factor, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x' or 'y'")
    result = cv.transform(points.astype(np.float64).reshape(-1, 1, 2), shear_matrix.reshape(2, 2)).reshape(-1, 2)
    print("Result:")
    print(result)
    return result
#univeral function
def transform(points, transformation_matrix):
    result= np.dot(points, transformation_matrix.T)
    print("Result:")
    print(result)
    return result
def transform_cv(points, transformation_matrix):
    result = cv.transform(points.astype(np.float64).reshape(-1, 1, 2), transformation_matrix.reshape(2, 2)).reshape(-1, 2)
    print("Result:")
    print(result)
    return result
#rotate 3d
def rotate_3d(points, angle, axis):
    theta = np.radians(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x', 'y', or 'z'")
    result= np.dot(points, rotation_matrix.T)
    print("Result:")
    print(result)
    return result
#scale 3d
def scale_3d(points, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, scale_factor]
    ])
    return np.dot(points, scaling_matrix)

def plot_3d( modified_vertices, title_original, title_modified):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

   
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  
        [4, 5], [5, 6], [6, 7], [7, 4], 
        [0, 4], [1, 5], [2, 6], [3, 7] ]

   

    if modified_vertices is not None:
        for edge in edges:
            ax.plot3D(*zip(modified_vertices[edge[0]], modified_vertices[edge[1]]), color='red')

        ax.set_title(title_modified)
    else:
        ax.set_title(title_original)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show() 
    
def plot_original_and_modified(original, modified, title_original, title_modified, shear_axis=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].plot(original[:, 0], original[:, 1], 'b-o')
    ax[0].set_title(title_original)
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].grid()

    if modified is not None:
        ax[1].plot(modified[:, 0], modified[:, 1], 'g-o')
        ax[1].set_title(title_modified)
        ax[1].set_xlim(-4, 4)
        ax[1].set_ylim(-4, 4)
        
        # Check if shear axis is specified and modify the axis labels accordingly
        if shear_axis == 'x':
            ax[1].set_xlabel('X-Shear')
            ax[1].set_ylabel('Y')
        elif shear_axis == 'y':
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y-Shear')
        else:
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')

        ax[1].grid()
    
    plt.show()

def plot_vectors(original, modified, title_original, title_modified):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    for i in range(0, len(original), 2):
        ax[0].quiver(original[i, 0], original[i, 1], original[i+1, 0], original[i+1, 1], angles='xy', scale_units='xy', scale=1, color='r')
    ax[0].set_title(title_original)
    ax[0].set_xlim(-4, 8)
    ax[0].set_ylim(-4, 8)
    ax[0].grid()

    for i in range(0, len(modified), 2):
        ax[1].quiver(modified[i, 0], modified[i, 1], modified[i+1, 0], modified[i+1, 1], angles='xy', scale_units='xy', scale=1, color='g')
    ax[1].set_title(title_modified)
    ax[1].set_xlim(-4, 8)
    ax[1].set_ylim(-4, 8)
    ax[1].grid()
    
    plt.show()


def rotate_image(image, angle, axis='z'):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
   
    if axis == 'x':
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    elif axis == 'y':
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    elif axis == 'z':
       
        rotation_matrix = None
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    
    if rotation_matrix is not None:
        rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))
    else:
       
        rotated_image = cv.rotate(image, cv.ROTATE_3D)

    return rotated_image

def scale_image(image, scale_factor):
    return cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
  
image = cv.imread('grid.jpg')

# Create Object 1
triangle = np.array([
    [0, 0],
    [1, 0],
    [0.7, 1],
    [0, 0]
])

# Create Object 2
vectors = np.array([
    [0, 0], [1, 1],
    [0, 0], [3, 2],
    [0, 0], [0, -1]
])
# Create 3d object(cube)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  
    [4, 5], [5, 6], [6, 7], [7, 4], 
    [0, 4], [1, 5], [2, 6], [3, 7] ]

while True:
    print("Menu:")
    print("1. Rotate object1")
    print("2. Scale object1")
    print("3. Rotate object2")
    print("4. Scale object2")
    print("5. Reflect object1")
    print("6. Reflect object2")
    print("7. Shear object1")
    print("8. Shear object2")
    print("9. Custom transformation on object1")
    print("10. Custom transformation on object2")
    print("11. Rotate 3d")
    print("12. Scale 3d")
    print("13. Rotate image")
    print("14. Scale image")
    print("15. Exit")
    
    choice = input("Choose an option: ")

    if choice == '1':
        angle = float(input("Enter the rotation angle (in degrees): "))
        rotated_triangle = rotate(triangle, angle)
        rotated_triangle_1 = rotate_cv(triangle, angle)
        plot_original_and_modified(triangle, rotated_triangle, 'Original Triangle', 'Rotated Triangle', None)
        plot_original_and_modified(triangle, rotated_triangle_1, 'Original Triangle', 'Rotated Triangle', None)
    elif choice == '2':
        scale_factor = float(input("Enter the scaling factor: "))
        scaled_triangle = scale(triangle, scale_factor)
        scaled_triangle_1 = scale_cv(triangle, scale_factor)
        plot_original_and_modified(triangle, scaled_triangle, 'Original Triangle', 'Scaled Triangle', None)
        plot_original_and_modified(triangle, scaled_triangle_1, 'Original Triangle', 'Scaled Triangle', None)
    elif choice == '3':
        angle = float(input("Enter the rotation angle (in degrees): "))
        rotated_vectors = rotate(vectors, angle)
        rotated_vectors_1 = rotate_cv(vectors, angle)
        plot_vectors(vectors, rotated_vectors, 'Original Vectors', 'Rotated Vectors')
        plot_vectors(vectors, rotated_vectors_1, 'Original Vectors', 'Rotated Vectors')
    elif choice == '4':
        scale_factor = float(input("Enter the scaling factor: "))
        scaled_vectors = scale(vectors, scale_factor)
        scaled_vectors_1 = scale_cv(vectors, scale_factor)
        plot_vectors(vectors, scaled_vectors, 'Original Vectors', 'Scaled Vectors')
        plot_vectors(vectors, scaled_vectors_1, 'Original Vectors', 'Scaled Vectors')
    elif choice == '5':
        axis = input("Enter the axis of reflection ('x' or 'y'): ").strip().lower()
        reflected_triangle = reflect(triangle, axis)
        reflected_triangle_1 = reflect_cv(triangle, axis)
        plot_original_and_modified(triangle, reflected_triangle, 'Original Triangle', 'Reflected Triangle', None)
        plot_original_and_modified(triangle, reflected_triangle_1, 'Original Triangle', 'Reflected Triangle', None)
    elif choice == '6':
        axis = input("Enter the axis of reflection ('x' or 'y'): ").strip().lower()
        reflected_vectors = reflect(vectors, axis)
        reflected_vectors_1 = reflect_cv(vectors, axis)
        plot_vectors(vectors, reflected_vectors, 'Original Vectors', 'Reflected Vectors')
        plot_vectors(vectors, reflected_vectors_1, 'Original Vectors', 'Reflected Vectors')
    elif choice == '7':
        axis = input("Enter the axis of shearing ('x' or 'y'): ").strip().lower()
        shear_factor = float(input("Enter the shearing factor: "))
        sheared_triangle = shear(triangle, axis, shear_factor)
        plot_original_and_modified(triangle, sheared_triangle, 'Original Triangle', 'Sheared Triangle', axis)

    elif choice == '8':
        axis = input("Enter the axis of shearing ('x' or 'y'): ").strip().lower()
        shear_factor = float(input("Enter the shearing factor: "))
        sheared_vectors = shear(vectors, axis, shear_factor)
        plot_vectors(vectors, sheared_vectors, 'Original Vectors', 'Sheared Vectors')

    elif choice == '9':
        custom_matrix = np.array([
            [float(input("Enter value for a11: ")), float(input("Enter value for a12: "))],
            [float(input("Enter value for a21: ")), float(input("Enter value for a22: "))]
        ])
        custom_transformed_triangle = transform(triangle, custom_matrix)
        custom_transformed_triangle_1 = transform_cv(triangle, custom_matrix)
        plot_original_and_modified(triangle, custom_transformed_triangle, 'Original Triangle', 'Custom Transformed Triangle', None)
        plot_original_and_modified(triangle, custom_transformed_triangle_1, 'Original Triangle', 'Custom Transformed Triangle', None)
    elif choice == '10':
        custom_matrix = np.array([
            [float(input("Enter value for a11: ")), float(input("Enter value for a12: "))],
            [float(input("Enter value for a21: ")), float(input("Enter value for a22: "))]
        ])
        custom_transformed_vectors = transform(vectors, custom_matrix)
        custom_transformed_vectors_1 = transform_cv(vectors, custom_matrix)
        plot_vectors(vectors, custom_transformed_vectors, 'Original Vectors', 'Custom Transformed Vectors')
        plot_vectors(vectors, custom_transformed_vectors_1, 'Original Vectors', 'Custom Transformed Vectors')
    elif choice == '11':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for edge in edges:
            ax.plot3D(*vertices[edge].T, color='blue')
        axis = input("Enter the axis of rotation ('x', 'y', or 'z'): ").strip().lower()
        angle = float(input("Enter the rotation angle (in degrees): "))
        rotated_vertices = rotate_3d(vertices, angle, axis)
        plot_3d( rotated_vertices, 'Original Object', 'Rotated Object')
    elif choice == '12':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for edge in edges:
            ax.plot3D(*vertices[edge].T, color='blue')
        scale_factor = float(input("Enter the scale_factor: "))
        scaled_cube = scale_3d(vertices, scale_factor)
        plot_3d(scaled_cube, 'Original Object', 'Scaled Object')
    elif choice == '13':    
        angle = float(input("Enter the angle: "))
        axis = input("Enter the axis of reflection ('x', 'y', 'z'): ").strip().lower()  
        rotated_image = rotate_image(image, angle, axis)
        cv.imshow('Original Image', image)
        cv.imshow('Rotated Image', rotated_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif choice == '14':
        scale_factor = float(input("Enter the scaling factor: "))
        scaled_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
        
        cv.imshow('Original Image', image)
        cv.imshow('Scaled Image', scaled_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif choice == '15':
        print("Exiting the program.")
        break

    else:
        print("Invalid choice, please try again.")
