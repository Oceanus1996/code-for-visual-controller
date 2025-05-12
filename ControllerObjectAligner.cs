using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";
    public string objectPathHint = "";
    public GameObject controllerObject;

    private Vector3 targetPositionGlobal;
    private int qPressCount = 0;
    private int maxPresses = 5;

    private Transform targetTransform;
    private List<float> distanceLog = new List<float>();

    public IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，等待 Controller (right) 或 Controller (left) ...");

        GameObject targetController = null;

        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                string name = obj.name;
                if (name == "Controller (right)" || name == "Controller (left)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 成功绑定 controllerObject → " + targetController.name);
                break;
            }

            Debug.Log("⏳ 等待 Controller (right) 或 Controller (left) 中...");
            yield return new WaitForSeconds(0.5f);
        }

        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();
        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        targetTransform = target;
        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄初始坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 初始距离：" + distance.ToString("F3"));

        targetPositionGlobal = targetPos;

        StartCoroutine(RecordDistanceContinuously());
        StartCoroutine(MonitorUntilKeyPress());
    }

    IEnumerator RecordDistanceContinuously()
    {
        while (qPressCount < maxPresses)
        {
            if (controllerObject != null && targetTransform != null)
            {
                float d = Vector3.Distance(controllerObject.transform.position, targetTransform.position);
                distanceLog.Add(d);
            }
            yield return null;
        }
    }

    IEnumerator MonitorUntilKeyPress()
    {
        Debug.Log("⌨️ 最多可以按 Q 键 5 次，每次记录当前手柄位置");

        while (true)
        {
            if (Input.GetKeyDown(KeyCode.Q))
            {
                if (qPressCount < maxPresses)
                {
                    qPressCount++;
                    Vector3 pos = controllerObject.transform.position;
                    SaveFinalPosition(pos, qPressCount);
                    Debug.Log("✅ 第 " + qPressCount + " 次记录成功！");
                }
                else if (qPressCount == maxPresses)
                {
                    Debug.Log("⚠️ 已达最大记录次数（5），按 Q 不再记录");

                    if (distanceLog.Count > 0)
                    {
                        float minDistance = Mathf.Min(distanceLog.ToArray());
                        Debug.Log("📉 实时记录期间的最小距离为：" + minDistance.ToString("F4"));
                    }
                }
            }

            yield return null;
        }
    }

    void SaveFinalPosition(Vector3 controllerPos, int index)
    {
        string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";

        string objectNameSafe = objectName.Replace(",", "_");
        Vector3 targetPos = targetPositionGlobal;
        float distance = Vector3.Distance(controllerPos, targetPos);
        float minDistance = (distanceLog.Count > 0) ? Mathf.Min(distanceLog.ToArray()) : distance;

        string line = string.Format(
            "{0}_{1},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6},{9:F6}",
            objectNameSafe, index,
            targetPos.x, targetPos.y, targetPos.z,
            controllerPos.x, controllerPos.y, controllerPos.z,
            distance, minDistance
        );

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "object_name,target_x,target_y,target_z,controller_x,controller_y,controller_z,distance,mindis\n");
        }

        File.AppendAllText(csvPath, line + "\n");
    }

    string GetFullPath(GameObject obj)
    {
        string path = obj.name;
        Transform current = obj.transform;
        while (current.parent != null)
        {
            current = current.parent;
            path = current.name + "/" + path;
        }
        return path;
    }
}



/*using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";
    public string objectPathHint = "";
    public GameObject controllerObject;

    private Vector3 targetPositionGlobal;
    private int qPressCount = 0;
    private int maxPresses = 5;

    private Transform targetTransform;
    private List<float> distanceLog = new List<float>();

    IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，等待 Controller (right) ...");

        GameObject targetController = null;

        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name == "Controller (right)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 成功绑定 controllerObject → Controller (right)");
                break;
            }

            Debug.Log("⏳ 等待 Controller (right) 中...");
            yield return new WaitForSeconds(0.5f);
        }




        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();
        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        targetTransform = target;
        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄初始坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 初始距离：" + distance.ToString("F3"));

        targetPositionGlobal = targetPos;

        StartCoroutine(RecordDistanceContinuously());
        StartCoroutine(MonitorUntilKeyPress());
    }

    IEnumerator RecordDistanceContinuously()
    {
        while (qPressCount < maxPresses)
        {
            if (controllerObject != null && targetTransform != null)
            {
                float d = Vector3.Distance(controllerObject.transform.position, targetTransform.position);
                distanceLog.Add(d);
            }
            yield return null;
        }
    }

    IEnumerator MonitorUntilKeyPress()
    {
        Debug.Log("⌨️ 最多可以按 Q 键 5 次，每次记录当前手柄位置");

        while (true)
        {
            if (Input.GetKeyDown(KeyCode.Q))
            {
                if (qPressCount < maxPresses)
                {
                    qPressCount++;
                    Vector3 pos = controllerObject.transform.position;
                    SaveFinalPosition(pos, qPressCount);
                    Debug.Log("✅ 第 " + qPressCount + " 次记录成功！");
                }
                else if (qPressCount == maxPresses)
                {
                    Debug.Log("⚠️ 已达最大记录次数（5），按 Q 不再记录");

                    if (distanceLog.Count > 0)
                    {
                        float minDistance = Mathf.Min(distanceLog.ToArray());
                        Debug.Log("📉 实时记录期间的最小距离为：" + minDistance.ToString("F4"));
                    }
                }
            }

            yield return null;
        }
    }

    void SaveFinalPosition(Vector3 controllerPos, int index)
    {
        string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";

        string objectNameSafe = objectName.Replace(",", "_");
        Vector3 targetPos = targetPositionGlobal;
        float distance = Vector3.Distance(controllerPos, targetPos);
        float minDistance = (distanceLog.Count > 0) ? Mathf.Min(distanceLog.ToArray()) : distance;

        string line = string.Format(
            "{0}_{1},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6},{9:F6}",
            objectNameSafe, index,
            targetPos.x, targetPos.y, targetPos.z,
            controllerPos.x, controllerPos.y, controllerPos.z,
            distance, minDistance
        );

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "object_name,target_x,target_y,target_z,controller_x,controller_y,controller_z,distance,mindis\n");
        }

        File.AppendAllText(csvPath, line + "\n");
    }

    string GetFullPath(GameObject obj)
    {
        string path = obj.name;
        Transform current = obj.transform;
        while (current.parent != null)
        {
            current = current.parent;
            path = current.name + "/" + path;
        }
        return path;
    }
}*/


//q五次+输出期间最小距离                      
/*using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";
    public string objectPathHint = "";
    public GameObject controllerObject;

    private Vector3 targetPositionGlobal;
    private int qPressCount = 0;
    private int maxPresses = 5;

    private Transform targetTransform;
    private List<float> distanceLog = new List<float>();

    IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，等待 Controller (left) ...");

        GameObject targetController = null;

        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name == "Controller (left)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 成功绑定 controllerObject → Controller (left)");
                break;
            }

            Debug.Log("⏳ 等待 Controller (left) 中...");
            yield return new WaitForSeconds(0.5f);
        }

        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();
        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        targetTransform = target;
        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄初始坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 初始距离：" + distance.ToString("F3"));

        targetPositionGlobal = targetPos;

        StartCoroutine(RecordDistanceContinuously());
        StartCoroutine(MonitorUntilKeyPress());
    }

    IEnumerator RecordDistanceContinuously()
    {
        while (qPressCount < maxPresses)
        {
            if (controllerObject != null && targetTransform != null)
            {
                float d = Vector3.Distance(controllerObject.transform.position, targetTransform.position);
                distanceLog.Add(d);
            }
            yield return null;
        }
    }

    IEnumerator MonitorUntilKeyPress()
    {
        Debug.Log("⌨️ 最多可以按 Q 键 5 次，每次记录当前手柄位置");

        while (true)
        {
            if (Input.GetKeyDown(KeyCode.Q))
            {
                if (qPressCount < maxPresses)
                {
                    qPressCount++;
                    Vector3 pos = controllerObject.transform.position;
                    SaveFinalPosition(pos, qPressCount);
                    Debug.Log("✅ 第 " + qPressCount + " 次记录成功！");
                }*/
/*else
{
    Debug.Log("⚠️ 已达最大记录次数（5），按 Q 不再记录");

    // 输出最小距离
    if (distanceLog.Count > 0)
    {
        float minDistance = Mathf.Min(distanceLog.ToArray());
        Debug.Log("📉 实时记录期间的最小距离为：" + minDistance.ToString("F4"));

        // 追加最小值行到 CSV
        string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";
        string minLine = string.Format("MinDistance_Total,,,,,,,{0:F6}", minDistance);
        File.AppendAllText(csvPath, minLine + "\n");
        Debug.Log("📝 最小距离已写入 CSV 末尾！");
    }
}*/
/*        else if (qPressCount == maxPresses)  // ✅ 只触发一次！
        {
            Debug.Log("⚠️ 已达最大记录次数（5），按 Q 不再记录");

            if (distanceLog.Count > 0)
            {
                float minDistance = Mathf.Min(distanceLog.ToArray());
                Debug.Log("📉 实时记录期间的最小距离为：" + minDistance.ToString("F4"));

                string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";
                string minLine = string.Format("MinDistance_Total,,,,,,,{0:F6}", minDistance);
                File.AppendAllText(csvPath, minLine + "\n");
                Debug.Log("📝 最小距离已写入 CSV 末尾！");
            }
        }

    }

    yield return null;
}
}

void SaveFinalPosition(Vector3 controllerPos, int index)
{
string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";

string objectNameSafe = objectName.Replace(",", "_");
Vector3 targetPos = targetPositionGlobal;
float distance = Vector3.Distance(controllerPos, targetPos);

string line = string.Format(
    "{0}_{1},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6}",
    objectNameSafe, index,
    targetPos.x, targetPos.y, targetPos.z,
    controllerPos.x, controllerPos.y, controllerPos.z,
    distance
);

if (!File.Exists(csvPath))
{
    File.AppendAllText(csvPath, "object_name,target_x,target_y,target_z,controller_x,controller_y,controller_z,distance\n");
}

File.AppendAllText(csvPath, line + "\n");
}

string GetFullPath(GameObject obj)
{
string path = obj.name;
Transform current = obj.transform;
while (current.parent != null)
{
    current = current.parent;
    path = current.name + "/" + path;
}
return path;
}
}

*/


/*using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";
    public string objectPathHint = "";
    public GameObject controllerObject;

    IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，持续等待 SteamVR 手柄 Controller (right)...");

        float startTime = Time.time;
        GameObject targetController = null;

        // 无限等到 Controller (right) 出现
        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name == "Controller (right)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 已绑定 controllerObject → Controller (right)");
                break;
            }

            Debug.Log("⏳ 等待中：还未找到 Controller (right)...");
            yield return new WaitForSeconds(0.5f);
        }

        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();

        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        // 输出并保存信息
        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 三维距离：" + distance.ToString("F3"));

        string savePath = Application.dataPath + "/target_pos.txt";
        File.WriteAllText(savePath, targetPos.x + "," + targetPos.y + "," + targetPos.z);
        Debug.Log("📁 目标坐标已保存到：" + savePath);


        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.transform.localScale = Vector3.one * 0.05f; // 小球尺寸
        marker.transform.position = controllerObject.transform.position;
        marker.GetComponent<Renderer>().material.color = Color.red;

    }

    string GetFullPath(GameObject obj)
    {
        string path = obj.name;
        Transform current = obj.transform;
        while (current.parent != null)
        {
            current = current.parent;
            path = current.name + "/" + path;
        }
        return path;
    }
}
*/


//一次q版本
/*using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";         // 目标物体名称
    public string objectPathHint = "";         // 可选 disambiguate 提示
    public GameObject controllerObject;        // 自动绑定 Controller (right)

    private Vector3 targetPositionGlobal;      // 保存目标坐标用于后续比较

    IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，等待 Controller (right) ...");

        GameObject targetController = null;

        // 无限等待直到 Controller (right) 出现
        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name == "Controller (right)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 成功绑定 controllerObject → Controller (right)");
                break;
            }

            Debug.Log("⏳ 等待 Controller (right) 中...");
            yield return new WaitForSeconds(0.5f);
        }

        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();
        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄初始坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 初始距离：" + distance.ToString("F3"));

        // 保存目标坐标
       *//* string savePath = Application.dataPath + "/target_pos.txt";
        string targetStr = string.Format("{0:F6},{1:F6},{2:F6}", targetPos.x, targetPos.y, targetPos.z);
        File.WriteAllText(savePath, targetStr, System.Text.Encoding.UTF8);
        Debug.Log("📄 已保存目标坐标：" + targetStr);*//*

        // 保存目标位置到全局变量
        targetPositionGlobal = targetPos;

        // 开始监听按键
        StartCoroutine(MonitorUntilKeyPress());
    }

    IEnumerator MonitorUntilKeyPress()
    {
        Debug.Log("⌨️ 等待你按下 Q 键来保存当前手柄坐标...");

        while (true)
        {
            if (Input.GetKeyDown(KeyCode.Q))
            {
                Vector3 pos = controllerObject.transform.position;
                SaveFinalPosition(pos);
                Debug.Log("✅ 已按下 Q，虚拟手柄坐标保存成功！");
                break;
            }

            yield return null;
        }
    }

    void SaveFinalPosition(Vector3 controllerPos)
    {
       *//* string cPath = Application.dataPath + "/controller_pos.txt";
        string cStr = string.Format("{0:F6},{1:F6},{2:F6}", controllerPos.x, controllerPos.y, controllerPos.z);
        File.WriteAllText(cPath, cStr, System.Text.Encoding.UTF8);
        Debug.Log("📁 手柄最终坐标已保存至：" + cStr);*//*

        // === 追加 CSV 记录 ===
        //string csvPath = Application.dataPath + "/result_log.csv";
        string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";

        string objectNameSafe = objectName.Replace(",", "_");
        Vector3 targetPos = targetPositionGlobal;
        float distance = Vector3.Distance(controllerPos, targetPos);

        string line = string.Format(
            "{0},{1:F6},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6}",
            objectNameSafe,
            targetPos.x, targetPos.y, targetPos.z,
            controllerPos.x, controllerPos.y, controllerPos.z,
            distance
        );

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "object_name,target_x,target_y,target_z,controller_x,controller_y,controller_z,distance\n");
        }

        File.AppendAllText(csvPath, line + "\n");
        Debug.Log("📝 CSV 已记录：\n" + line);
    }

    string GetFullPath(GameObject obj)
    {
        string path = obj.name;
        Transform current = obj.transform;
        while (current.parent != null)
        {
            current = current.parent;
            path = current.name + "/" + path;
        }
        return path;
    }
}
*/


//最多q五次版
/*
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ControllerObjectAligner : MonoBehaviour
{
    public string objectName = "Cube";         // 目标物体名称
    public string objectPathHint = "";         // 可选路径提示
    public GameObject controllerObject;        // 自动绑定 Controller (right)

    private Vector3 targetPositionGlobal;
    private int qPressCount = 0;
    private int maxPresses = 5;

    IEnumerator Start()
    {
        Debug.Log("▶ 脚本启动，等待 Controller (right) ...");

        GameObject targetController = null;

        // 无限等待直到 Controller (right) 出现
        while (targetController == null)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name == "Controller (right)")
                {
                    targetController = obj;
                    break;
                }
            }

            if (targetController != null)
            {
                controllerObject = targetController;
                Debug.Log("✅ 成功绑定 controllerObject → Controller (right)");
                break;
            }

            Debug.Log("⏳ 等待 Controller (right) 中...");
            yield return new WaitForSeconds(0.5f);
        }

        // 查找目标物体
        Transform[] all = GameObject.FindObjectsOfType<Transform>();
        List<Transform> matches = new List<Transform>();
        foreach (Transform t in all)
        {
            if (t.name == objectName)
                matches.Add(t);
        }

        if (matches.Count == 0)
        {
            Debug.LogWarning("❌ 没找到目标物体名：" + objectName);
            yield break;
        }

        Transform target = null;
        if (matches.Count == 1)
        {
            target = matches[0];
        }
        else
        {
            foreach (Transform t in matches)
            {
                string path = GetFullPath(t.gameObject);
                if (path.EndsWith(objectPathHint))
                {
                    target = t;
                    break;
                }
            }

            if (target == null)
            {
                Debug.LogWarning("⚠️ 有多个同名物体，请填写 objectPathHint。路径如下：");
                foreach (Transform t in matches)
                    Debug.Log("🔎 " + GetFullPath(t.gameObject));
                yield break;
            }
        }

        Vector3 controllerPos = controllerObject.transform.position;
        Vector3 targetPos = target.position;
        float distance = Vector3.Distance(controllerPos, targetPos);

        Debug.Log("🎮 虚拟手柄初始坐标：" + controllerPos);
        Debug.Log("🎯 目标物体坐标：" + targetPos);
        Debug.Log("📏 初始距离：" + distance.ToString("F3"));

        // 保存目标位置
        targetPositionGlobal = targetPos;

        // 开始监听 Q 键
        StartCoroutine(MonitorUntilKeyPress());
    }

    IEnumerator MonitorUntilKeyPress()
    {
        Debug.Log("⌨️ 最多可以按 Q 键 5 次，每次记录当前手柄位置");

        while (true)
        {
            if (Input.GetKeyDown(KeyCode.Q))
            {
                if (qPressCount < maxPresses)
                {
                    qPressCount++;
                    Vector3 pos = controllerObject.transform.position;
                    SaveFinalPosition(pos, qPressCount);
                    //Debug.Log($"✅ 第 {qPressCount} 次记录成功！");
                    Debug.Log("✅ 第 " + qPressCount + " 次记录成功！");

                }
                else
                {
                    Debug.Log("⚠️ 已达最大记录次数（5），按 Q 不再记录");
                }
            }

            yield return null;
        }
    }

    void SaveFinalPosition(Vector3 controllerPos, int index)
    {
        // 你指定的 CSV 输出路径
        string csvPath = @"D:\VRproject\SpyVR-master\VR_Project\Assets\positiondata\result_log.csv";

        string objectNameSafe = objectName.Replace(",", "_");
        Vector3 targetPos = targetPositionGlobal;
        float distance = Vector3.Distance(controllerPos, targetPos);

        string line = string.Format(
            "{0}_{1},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6}",
            objectNameSafe, index,
            targetPos.x, targetPos.y, targetPos.z,
            controllerPos.x, controllerPos.y, controllerPos.z,
            distance
        );

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "object_name,target_x,target_y,target_z,controller_x,controller_y,controller_z,distance\n");
        }

        File.AppendAllText(csvPath, line + "\n");
        Debug.Log("📝 已写入 CSV:\n" + line);
    }

    string GetFullPath(GameObject obj)
    {
        string path = obj.name;
        Transform current = obj.transform;
        while (current.parent != null)
        {
            current = current.parent;
            path = current.name + "/" + path;
        }
        return path;
    }
}*/
