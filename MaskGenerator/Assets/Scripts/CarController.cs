using UnityEngine;
using UnityEngine.Serialization;

public class CarController : MonoBehaviour
{
    [Header("Wheel Colliders")] public WheelCollider frontLeftWheel;

    public WheelCollider frontRightWheel;
    public WheelCollider rearLeftWheel;
    public WheelCollider rearRightWheel;

    [Header("Wheel Transforms")] public Transform frontLeftTransform;

    public Transform frontRightTransform;
    public Transform rearLeftTransform;
    public Transform rearRightTransform;

    [Header("Car Settings")] public float motorForce = 1500f;

    public float passiveBrakeForce = 500f;
    public float directionChangeBrakeForce = 3000f;
    public float maxSteerAngle = 30f;
    public float maxSpeed = 50f;
    public float minSpeed = -50f;
    public float forwardFrictionStiffness = 2f;
    public Rigidbody carRigidbody;

    public float sidewaysFrictionStiffness = 2f;
    // private bool isBraking;

    private float _currentBrakeForce;
    private float _currentMotorForce;

    private float _currentSteerAngle;

    public void Reset()
    {
        _currentSteerAngle = 0f;
        _currentBrakeForce = 0f;
        _currentMotorForce = 0f;
        // isBraking = false;
        carRigidbody.linearVelocity = Vector3.zero;
        carRigidbody.angularVelocity = Vector3.zero;

        frontLeftWheel.steerAngle = _currentSteerAngle;
        frontRightWheel.steerAngle = _currentSteerAngle;

        frontLeftWheel.motorTorque = 0f;
        frontRightWheel.motorTorque = 0f;
        rearLeftWheel.motorTorque = 0f;
        rearRightWheel.motorTorque = 0f;
    }

    private void Start()
    {
        AdjustWheelGrip(frontLeftWheel);
        AdjustWheelGrip(frontRightWheel);
        AdjustWheelGrip(rearLeftWheel);
        AdjustWheelGrip(rearRightWheel);
    }

    private void FixedUpdate()
    {
        LimitSpeed();
        HandleMotor();
        UpdateWheels();
    }

    private void AdjustWheelGrip(WheelCollider wheel)
    {
        WheelFrictionCurve forwardFriction = wheel.forwardFriction;
        forwardFriction.stiffness = forwardFrictionStiffness;
        wheel.forwardFriction = forwardFriction;

        WheelFrictionCurve sidewaysFriction = wheel.sidewaysFriction;
        sidewaysFriction.stiffness = sidewaysFrictionStiffness;
        wheel.sidewaysFriction = sidewaysFriction;
    }

    public void Move(float force)
    {
        _currentMotorForce = force * motorForce;
    }

    private void HandleMotor()
    {
        float currentSpeed = carRigidbody.linearVelocity.magnitude *
                             Mathf.Sign(Vector3.Dot(carRigidbody.linearVelocity, transform.forward));
        bool shouldBrake = (_currentMotorForce > 0 && currentSpeed < 0) || (_currentMotorForce < 0 && currentSpeed > 0);

        if (shouldBrake)
        {
            _currentBrakeForce = directionChangeBrakeForce;
            frontLeftWheel.motorTorque = 0f;
            frontRightWheel.motorTorque = 0f;
            rearLeftWheel.motorTorque = 0f;
            rearRightWheel.motorTorque = 0f;
        }
        else if (Mathf.Abs(_currentMotorForce) > 0.01f)
        {
            frontLeftWheel.motorTorque = _currentMotorForce;
            frontRightWheel.motorTorque = _currentMotorForce;
            rearLeftWheel.motorTorque = _currentMotorForce;
            rearRightWheel.motorTorque = _currentMotorForce;

            _currentBrakeForce = 0f;
        }
        else
        {
            _currentBrakeForce = passiveBrakeForce;
        }

        ApplyBraking();
    }

    private void ApplyBraking()
    {
        frontLeftWheel.brakeTorque = _currentBrakeForce;
        frontRightWheel.brakeTorque = _currentBrakeForce;
        rearLeftWheel.brakeTorque = _currentBrakeForce;
        rearRightWheel.brakeTorque = _currentBrakeForce;
    }

    public void Turn(float direction)
    {
        // float speedFactor = Mathf.Clamp01(carRigidbody.linearVelocity.magnitude / maxSpeed);
        // float dynamicSteerAngle = maxSteerAngle * (1 - speedFactor);

        _currentSteerAngle = maxSteerAngle * direction / 2;
        // currentSteerAngle = dynamicSteerAngle * direction;
        // frontLeftWheel.steerAngle = _currentSteerAngle;
        // frontRightWheel.steerAngle = _currentSteerAngle;
    }

    private void UpdateWheels()
    {
        frontLeftWheel.steerAngle = _currentSteerAngle;
        frontRightWheel.steerAngle = _currentSteerAngle;
        UpdateWheelPose(frontLeftWheel, frontLeftTransform);
        UpdateWheelPose(frontRightWheel, frontRightTransform);
        UpdateWheelPose(rearLeftWheel, rearLeftTransform);
        UpdateWheelPose(rearRightWheel, rearRightTransform);
    }

    private void UpdateWheelPose(WheelCollider wheelCollider, Transform wheelTransform)
    {
        wheelCollider.GetWorldPose(out Vector3 pos, out Quaternion rot);
        wheelTransform.position = pos;
        wheelTransform.rotation = rot;
    }

    public float Speed()
    {
        return carRigidbody.linearVelocity.magnitude;
    }

    public float Steering()
    {
        // float speedFactor = Mathf.Clamp01(carRigidbody.linearVelocity.magnitude / maxSpeed);
        // float dynamicSteerAngle = maxSteerAngle * (1 - speedFactor);

        // return currentSteerAngle / dynamicSteerAngle;
        return _currentSteerAngle / maxSteerAngle;
    }

    private void LimitSpeed()
    {
        if (Speed() > maxSpeed || Speed() < minSpeed) _currentMotorForce = 0f;
    }
}