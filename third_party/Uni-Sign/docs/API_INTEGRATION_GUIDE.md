# ĐẶC TẢ TÍCH HỢP DỊCH VỤ AI UNISIGN (SIGN2TEXT)
## Dành cho Team Backend và Frontend

Tài liệu này hướng dẫn chi tiết cách kết nối, tích hợp và sử dụng dịch vụ AI nhận diện ngôn ngữ ký hiệu Tiếng Việt (**Uni-Sign ISLR Inference Service**) cho đội ngũ phát triển Backend và Frontend.

---

## 1. TỔNG QUAN KIẾN TRÚC & MÔ HÌNH KẾT NỐI

Dịch vụ AI `Uni-Sign` được triển khai dưới dạng **Private Microservice** độc lập trên Docker container, xử lý nhận diện clip cử chỉ ký hiệu đơn lẻ (Isolated Sign Language Recognition - ISLR) và trả về nhãn từ ngữ Tiếng Việt tương ứng.

```text
┌─────────────────────────┐          Upload Clip (WebM/MP4)          ┌─────────────────────────┐
│                         │ ────────── (HTTPS / REST API) ─────────> │                         │
│   Frontend (Web / App)  │                                          │   Backend (Main API)    │
│                         │ <────────── Kết quả JSON ──────────────  │                         │
└─────────────────────────┘                                          └─────────────────────────┘
                                                                                  │
                                                                         Gọi Private AI API
                                                                       (http://127.0.0.1:56568)
                                                                                  │
                                                                                  ▼
                                                                     ┌─────────────────────────┐
                                                                     │   Uni-Sign AI Service   │
                                                                     │ (FastAPI + PyTorch GPU) │
                                                                     └─────────────────────────┘
```

> ⚠️ **LƯU Ý BẢO MẬT QUAN TRỌNG:**
> * **Frontend KHÔNG gọi trực tiếp vào Uni-Sign AI Service** để tránh lộ `SERVICE_API_KEY` và quản lý tải.
> * **Backend gọi trực tiếp** tới AI Service qua mạng nội bộ Docker/Localhost.

---

## 2. HƯỚNG DẪN DÀNH CHO TEAM BACKEND

### 2.1. Thông Tin Kết Nối Nội Bộ (Internal Configuration)

* **Internal Base URL (Chung Server/Localhost):** `http://127.0.0.1:56568`
* **Docker Network Alias (Trong Docker Network):** `http://docker-unisign-inference-1:8080`
* **Authentication Header:**  
  `Authorization: Bearer <SERVICE_API_KEY>`

---

### 2.2. Danh Sách APIs Chi Tiết

#### 1. Kiểm tra trạng thái sẵn sàng (Healthcheck Readiness)
Trước khi gửi yêu cầu nhận diện, Backend có thể kiểm tra xem mô hình AI trên GPU đã được nạp xong chưa.

* **Endpoint:** `GET /readyz`
* **Auth:** Không yêu cầu
* **Response `200 OK` (Sẵn sàng):**
  ```json
  {
    "status": "ready",
    "model": {
      "id": "cosign-vi-islr",
      "version": "2026-08-04-r1",
      "label_count": 30,
      "max_frames": 64,
      "pose_mode": "performance",
      "dtype": "float32"
    }
  }
  ```
* **Response `503 Service Unavailable`:** Dịch vụ đang khởi động hoặc mô hình chưa nạp xong.

---

#### 2. Nhận diện video clip ký hiệu (Sign Recognition)

* **Endpoint:** `POST /v1/predictions`
* **Auth Header:** `Authorization: Bearer <SERVICE_API_KEY>`
* **Content-Type:** `multipart/form-data`
* **Request Fields:**

| Field Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `video` | `File` | **Bắt buộc** | File video clip (`MP4`, `WebM`, `MOV`, `AVI`), dung lượng <= 20MB, thời lượng 3-8s. |
| `top_k` | `Integer` | Tùy chọn | Số lượng nhãn gợi ý tốt nhất muốn lấy (từ `1` đến `5`, mặc định `5`). |
| `client_capture_id` | `String` | Tùy chọn | ID phiên làm việc từ client (nếu có, tối đa 128 ký tự). |

* **Ví dụ cURL:**
  ```bash
  curl -s -X POST "http://127.0.0.1:56568/v1/predictions" \
    -H "Authorization: Bearer <SERVICE_API_KEY>" \
    -F "video=@sample_sign.mp4;type=video/mp4" \
    -F "top_k=5" \
    -F "client_capture_id=sess-12345"
  ```

* **Response `200 OK` (Thành công):**
  ```json
  {
    "status": "OK",
    "request_id": "9bfcbf42-8243-4a62-8b87-8329d5ae0aa7",
    "created_at": "2026-08-06T10:50:00Z",
    "client_capture_id": "sess-12345",
    "prediction": {
      "top_1": "Chào",
      "top_k": [
        {"label": "Chào", "relative_score": 0.8542},
        {"label": "Bạn thân", "relative_score": 0.0811},
        {"label": "Chúng ta", "relative_score": 0.0325},
        {"label": "Học sinh", "relative_score": 0.0180},
        {"label": "Hôm nay", "relative_score": 0.0142}
      ]
    },
    "quality": {
      "person_detected": true,
      "person_coverage": 0.9421,
      "mean_hand_score": 0.6812,
      "pass": true
    },
    "timing_ms": {
      "video_decode": 45.2,
      "pose_extraction": 180.5,
      "model_inference": 32.1,
      "total": 257.8
    }
  }
  ```

---

#### 3. Lấy danh sách từ vựng hỗ trợ (Supported Vocabulary)

* **Endpoint:** `GET /v1/labels`
* **Auth Header:** `Authorization: Bearer <SERVICE_API_KEY>`
* **Response `200 OK`:**
  ```json
  {
    "labels": [
      "Ban ngày", "Ban đêm", "Bàn tay", "Bạn thân", "Bệnh viện", 
      "Chiều", "Chào", "Chân", "Chúng ta", "Chậm lại", 
      "Con gấu", "Cá", "Có thể", "Cơ thể", "Cứu", 
      "Dễ", "Hôm nay", "Họ", "Học sinh", "Khóc", 
      "Mua", "Mời vào", "Nghe", "Ngón tay", "Nhà", 
      "Nhìn", "Nhầm", "Nói", "Nặng", "Ăn"
    ],
    "model": {
      "id": "cosign-vi-islr",
      "version": "2026-08-04-r1",
      "label_count": 30
    }
  }
  ```

---

### 2.3. Mã Lỗi Chuẩn (Error Handling for Backend)

Khi gọi dịch vụ thất bại, HTTP Status Code và định dạng lỗi theo chuẩn `Problem Details` (`RFC 7807`) như sau:

```json
{
  "type": "https://unisign.internal/problems/no_person_detected",
  "title": "No person was detected in the video",
  "status": 400,
  "code": "NO_PERSON_DETECTED",
  "request_id": "9bfcbf42-8243-4a62-8b87-8329d5ae0aa7",
  "retryable": false
}
```

| HTTP Status | Error Code | Nguyên nhân & Hướng xử lý |
| :--- | :--- | :--- |
| `400` | `INVALID_MULTIPART` | File upload bị lỗi hoặc không đúng định dạng multipart. |
| `400` | `VIDEO_TOO_LONG` | Video vượt quá 8 giây. Nhắc người dùng quay clip ngắn hơn (3-5s). |
| `400` | `NO_PERSON_DETECTED` | Không phát hiện được dáng người trong video. Nhắc người dùng đứng rõ trước camera. |
| `401` | `UNAUTHORIZED` | Sai khóa `SERVICE_API_KEY`. Kiểm tra lại cấu hình Backend. |
| `413` | `FILE_TOO_LARGE` | Dung lượng video vượt quá 20MB. |
| `429` | `GPU_QUEUE_FULL` | Hàng đợi GPU đang bận (`retryable: true`). Cho phép gửi lại yêu cầu sau 1-2 giây. |
| `503` | `MODEL_NOT_READY` | Mô hình chưa sẵn sàng. |

---

### 2.4. Code Snippet Ví Dụ Cho Backend

#### Node.js / Express (Dùng Axios & FormData):
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function predictSign(videoPath) {
  const form = new FormData();
  form.append('video', fs.createReadStream(videoPath));
  form.append('top_k', 5);

  try {
    const response = await axios.post('http://127.0.0.1:56568/v1/predictions', form, {
      headers: {
        ...form.getHeaders(),
        'Authorization': `Bearer ${process.env.SERVICE_API_KEY || '<SERVICE_API_KEY>'}`
      },
      timeout: 10000
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      console.error("AI Service Error:", error.response.data);
    }
    throw error;
  }
}
```

#### Python / FastAPI (Dùng httpx):
```python
import os
import httpx

async def predict_sign(video_bytes: bytes, filename: str = "clip.webm"):
    url = "http://127.0.0.1:56568/v1/predictions"
    api_key = os.getenv("SERVICE_API_KEY", "<SERVICE_API_KEY>")
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"video": (filename, video_bytes, "video/webm")}
    data = {"top_k": 5}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
```

---

## 3. HƯỚNG DẪN DÀNH CHO TEAM FRONTEND

### 3.1. Quy Trình Thu Thập Video Trên Trình Duyệt (Webcam Capture)

Sử dụng HTML5 `getUserMedia` và `MediaRecorder` để ghi lại thao tác ký hiệu của người dùng:

1. **Định dạng ghi hình (MimeType):** `video/webm;codecs=vp8` hoặc `video/mp4`.
2. **Thời lượng clip tối ưu:** **3 đến 5 giây** (Bắt đầu quay -> Người dùng thực hiện 1 ký hiệu -> Dừng quay).
3. **Góc quay & Ánh sáng chuẩn:**
   * Giữ camera cố định, người ký hiệu đứng ở chính giữa.
   * Biên độ cử chỉ nằm gọn trong khung hình (đầu, ngực, hai tay).
4. **Gửi file:** Đẩy Blob/File video lên Backend API của bạn (chứ không gọi thẳng vào port `56568`).

---

### 3.2. Thiết Kế Giao Diện Trải Nghiệm Người Dùng (UI/UX Best Practices)

* **Màn hình chính:** 
  * Hiển thị kết quả **Top-1** chữ Tiếng Việt lớn và rõ ràng (Ví dụ: **"Chào"**).
* **Danh sách gợi ý (Top-5 Suggestions):**
  * Hiển thị danh sách 5 từ gợi ý kèm thanh tỷ lệ điểm tin cậy `relative_score`.
  * Cho phép người dùng bấm chọn từ đúng nếu Top-1 chưa chính xác.
* **Xử lý Thông báo Lỗi (User-facing Messages):**

| Mã lỗi từ API | Thông báo gợi ý hiển thị cho Người dùng |
| :--- | :--- |
| `NO_PERSON_DETECTED` | "Không tìm thấy người trong camera. Vui lòng đứng rõ trước camera và thử lại." |
| `VIDEO_TOO_LONG` | "Đoạn video quá dài. Vui lòng chỉ quay thao tác ký hiệu trong 3-5 giây." |
| `GPU_QUEUE_FULL` | "Hệ thống đang xử lý nhiều yêu cầu. Vui lòng thử lại sau vài giây." |
| `FILE_TOO_LARGE` | "Dung lượng video quá lớn. Vui lòng thử lại." |

---

## 4. TÓM TẮT CHECKLIST CHO DEV TEAM

- [ ] **Backend Dev**: Đã lưu URL `http://127.0.0.1:56568` và Token `SERVICE_API_KEY` vào file `.env` của Backend service.
- [ ] **Backend Dev**: Đã tạo route trung gian trong Backend để nhận video từ Frontend và forward sang `POST /v1/predictions`.
- [ ] **Frontend Dev**: Đã lập trình MediaRecorder giới hạn clip 3-5s và xử lý hiển thị danh sách kết quả Top-5.
