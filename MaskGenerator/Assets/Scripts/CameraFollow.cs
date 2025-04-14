using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;
    public float distance = 5.0f;
    public float height = 2.0f;

    private void LateUpdate()
    {
        Vector3 desiredPosition = target.position - target.forward * distance + Vector3.up * height;
        transform.position = desiredPosition;
        transform.LookAt(target.position + Vector3.up * 1.0f);
    }
}