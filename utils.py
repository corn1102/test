import numpy as np
import torch
from torchvision import transforms,models
import os
from PIL import Image
import random
from torch.utils import data
import math
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)
    
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


class ImageNetDataset(Dataset):
	def __init__(self, path, H=224, W=224):
		super(ImageNetDataset, self).__init__()
		self.H = H
		self.W = W
		self.root_dir = path
		self.transform = transforms.Compose([
			transforms.Resize((int(self.H ), int(self.W))),
			transforms.ToTensor(),

		])

		self.classes, self.class_to_idx = self._find_classes()
		
	def _find_classes(self):
		classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
		classes.sort()
		class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
		return classes, class_to_idx

	def _get_image_paths(self):
		image_paths = []
		for cls_name in self.classes:
			class_dir = os.path.join(self.root_dir, cls_name)
			for root, dirs, files in os.walk(class_dir):
				for file in files:
					if file.endswith((".jpg", ".JPEG", ".png")):
						image_paths.append(os.path.join(root, file))
		return image_paths

	def __len__(self):
		return len(self._get_image_paths())

	def __getitem__(self, idx):
		image_path = self._get_image_paths()[idx]
		image = Image.open(image_path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		label = self.class_to_idx[os.path.basename(os.path.dirname(image_path))]
		return image, label

def tensor_to_image(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = tensor.squeeze(0)
    image = inv_normalize(image)


    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image

def image_to_tensor(image_path, H = 224, W = 224):

    transform = transforms.Compose([
        transforms.Resize((int(H), int(W))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    image = Image.open(image_path).convert('RGB')

    transformed_image = transform(image)
    
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image

def normalize(inputs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    n_channels = len(mean)
    mean = torch.tensor(mean).reshape(1, n_channels, 1, 1).to(device)
    std = torch.tensor(std).reshape(1, n_channels, 1, 1).to(device)

    return (inputs - mean) / std

def inverse_normalize(inputs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    n_channels = len(mean)
    mean = torch.tensor(mean).reshape(1, n_channels, 1, 1).to(device)
    std = torch.tensor(std).reshape(1, n_channels, 1, 1).to(device)

    return inputs * std + mean

def preprocess_watermark(image_path, target_size, ratio):
    watermark = Image.open(image_path).convert('L')
    
    resize = transforms.Resize((int(target_size[0]*ratio), int(target_size[1]*ratio)))
    padding = transforms.Pad((0, 0, int(target_size[0]*(1-ratio)), int(target_size[1]*(1-ratio))))
    watermark = padding(resize(watermark))

    threshold = 128
    watermark = transforms.functional.to_tensor(watermark)
    watermark = torch.where(watermark > threshold/255, torch.tensor([1.0]), torch.tensor([0.0]))
   
    crop = transforms.CenterCrop(target_size)
    watermark = crop(watermark)
    watermark = watermark.squeeze(0)
    
    return watermark

def preprocess_watermark_mask(image_path, target_size, size_ratio, start_ratio):

    watermark = Image.open(image_path).convert('L')

    resize = transforms.Resize((int(target_size[0]*size_ratio), int(target_size[1]*size_ratio)))
    
    padding = transforms.Pad((int(target_size[0]*(start_ratio)), int(target_size[0]*(start_ratio)) , int(target_size[0]*(1 - size_ratio - start_ratio)), int(target_size[1]*(1 - size_ratio - start_ratio))))
    watermark = padding(resize(watermark))

    threshold = 128
    watermark = transforms.functional.to_tensor(watermark)
    watermark = torch.where(watermark > threshold/255, torch.tensor([1.0]), torch.tensor([0.0]))
    
    crop = transforms.CenterCrop(target_size)
    watermark = crop(watermark)
    watermark = watermark.squeeze(0)
    
    return watermark


def compute_NC(watermark1_tensor, watermark2_tensor):

    mean1 = torch.mean(watermark1_tensor)
    mean2 = torch.mean(watermark2_tensor)
    std1 = torch.std(watermark1_tensor)
    std2 = torch.std(watermark2_tensor)

    watermark1_normalized = (watermark1_tensor - mean1) / std1
    watermark2_normalized = (watermark2_tensor - mean2) / std2

    correlation_coefficient = torch.sum((watermark1_normalized - torch.mean(watermark1_normalized)) * \
                                        (watermark2_normalized - torch.mean(watermark2_normalized))) / \
                               (torch.sqrt(torch.sum((watermark1_normalized - torch.mean(watermark1_normalized)) ** 2)) * \
                                torch.sqrt(torch.sum((watermark2_normalized - torch.mean(watermark2_normalized)) ** 2)))

    NC_value = (correlation_coefficient + 1) / 2

    return NC_value.item() 

def psnr_2d(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def preprocess_ex_watermark(ex_watermark,target_size, ratio):

    target_size = (int(target_size[0]*ratio),int(target_size[1]*ratio))

    ex_watermark = ex_watermark[:target_size[0], :target_size[1]]

    return ex_watermark


def extract_watermark(watermarked_image, p, mask):
 
    watermarked_dct = dct_2d(watermarked_image)
    watermark = (watermarked_dct[0] % p < p / 2).sum(dim=0)
    watermark = (watermark >= 2).float()  
    watermark = watermark * mask[0,0,:,:]
    return watermark

def embed_watermark(images, watermark, p, mask):    
    images_dct = dct_2d(images)
    watermarked_dct = images_dct.clone().to(device)

    watermark_condition = watermark == 1 
    watermark_mask = watermark_condition.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    watermarked_dct[watermark_mask & mask.bool()] += (watermarked_dct[watermark_mask & mask.bool()] % p >= p / 2).float() * (p / 2)
    watermarked_dct[~watermark_mask & mask.bool()] += (watermarked_dct[~watermark_mask & mask.bool()] % p < p / 2).float() * (p / 2)
    watermarked_image = idct_2d(watermarked_dct)

    watermarked_image = torch.clamp(watermarked_image, min=0, max=1).detach()


    return watermarked_image


def Pixel_extract_watermark(watermarked_image, p, mask):
     
    watermarked_image = watermarked_image*255.0
    watermark = (watermarked_image[0] % p < p / 2).sum(dim=0)
    watermark = (watermark >= 2).float()  
    watermark = watermark * mask[0,0,:,:]
    return watermark

def Pixel_embed_watermark(images, watermark, p, mask):
       
    # images_dct = dct_2d(images)
    watermarked_image = images.clone().to(device) * 255.0

    watermark_condition = watermark == 1
    watermark_mask = watermark_condition.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    watermarked_image[watermark_mask & mask.bool()] += (watermarked_image[watermark_mask & mask.bool()] % p >= p / 2).float() * (p / 2)
    watermarked_image[~watermark_mask & mask.bool()] += (watermarked_image[~watermark_mask & mask.bool()] % p < p / 2).float() * (p / 2)

    watermarked_image = torch.clamp(watermarked_image/255.0, min=0, max=1).detach()

    return watermarked_image
    
classes = ['65', '970', '230', '809', '516', '57', '334', '415', '674', '332', '109', '286', '370', '757', '595', '147', '108', '23', '478', '517', '334', '173', '948', '727', '23', '846', '270', '167', '55', '858', '324', '573', '150', '981', '586', '887', '32', '398', '777', '74', '516', '756', '129', '198', '256', '725', '565', '167', '717', '394', '92', '29', '844', '591', '358', '468', '259', '994', '872', '588', '474', '183', '107', '46', '842', '390', '101', '887', '870', '841', '467', '149', '21', '476', '80', '424', '159', '275', '175', '461', '970', '160', '788', '58', '479', '498', '369', '28', '487', '50', '270', '383', '366', '780', '373', '705', '330', '142', '949', '349', '473', '159', '872', '878', '201', '906', '70', '486', '632', '608', '122', '720', '227', '686', '173', '959', '638', '646', '664', '645', '718', '483', '852', '392', '311', '457', '352', '22', '934', '283', '802', '553', '276', '236', '751', '343', '528', '328', '969', '558', '163', '328', '771', '726', '977', '875', '265', '686', '590', '975', '620', '637', '39', '115', '937', '272', '277', '763', '789', '646', '213', '493', '647', '504', '937', '687', '781', '666', '583', '158', '825', '212', '659', '257', '436', '196', '140', '248', '339', '230', '361', '544', '935', '638', '627', '289', '867', '272', '103', '584', '180', '703', '449', '771', '118', '396', '934', '16', '548', '993', '704', '457', '233', '401', '827', '376', '146', '606', '922', '516', '284', '889', '475', '978', '475', '984', '16', '77', '610', '254', '636', '662', '473', '213', '25', '463', '215', '173', '35', '741', '125', '787', '289', '425', '973', '1', '167', '121', '445', '702', '532', '366', '678', '764', '125', '349', '13', '179', '522', '493', '989', '720', '438', '660', '983', '533', '487', '27', '644', '750', '865', '1', '176', '694', '695', '798', '925', '413', '250', '970', '821', '421', '361', '920', '761', '27', '676', '92', '194', '897', '612', '610', '283', '881', '906', '899', '426', '554', '403', '826', '869', '730', '0', '866', '580', '888', '43', '64', '69', '176', '329', '469', '292', '991', '591', '346', '1', '607', '934', '784', '572', '389', '979', '654', '420', '390', '702', '24', '102', '949', '508', '361', '280', '65', '777', '359', '165', '21', '7', '525', '760', '938', '254', '733', '707', '463', '60', '887', '531', '380', '982', '305', '355', '503', '818', '495', '472', '293', '816', '195', '904', '475', '481', '431', '260', '130', '627', '460', '622', '696', '300', '37', '133', '637', '675', '465', '592', '741', '895', '91', '109', '582', '694', '546', '208', '488', '786', '959', '192', '834', '879', '649', '228', '621', '630', '107', '598', '420', '134', '133', '185', '471', '230', '974', '74', '76', '852', '383', '267', '419', '359', '484', '510', '33', '177', '935', '310', '998', '270', '598', '199', '998', '836', '14', '97', '856', '398', '319', '549', '92', '765', '412', '945', '160', '265', '638', '619', '722', '183', '674', '468', '468', '885', '675', '636', '196', '912', '721', '16', '199', '175', '775', '944', '350', '557', '361', '361', '594', '861', '257', '606', '734', '767', '746', '788', '346', '153', '739', '414', '915', '940', '152', '943', '849', '712', '100', '546', '744', '764', '141', '39', '993', '758', '190', '888', '18', '584', '341', '875', '359', '388', '894', '437', '987', '517', '372', '286', '662', '713', '915', '964', '146', '529', '416', '376', '147', '902', '26', '398', '175', '270', '335', '532', '540', '607', '495', '222', '801', '982', '304', '166', '56', '868', '448', '744', '567', '277', '298', '651', '377', '684', '832', '39', '219', '863', '868', '794', '80', '983', '269', '238', '498', '223', '521', '830', '260', '491', '896', '220', '680', '48', '542', '961', '820', '148', '114', '99', '143', '691', '796', '986', '346', '367', '939', '875', '625', '482', '464', '812', '705', '860', '466', '781', '499', '338', '488', '858', '795', '437', '11', '625', '965', '874', '928', '600', '86', '133', '149', '865', '480', '325', '499', '834', '421', '298', '900', '905', '184', '740', '258', '762', '295', '129', '240', '833', '471', '385', '899', '162', '288', '450', '850', '227', '273', '954', '965', '611', '643', '147', '290', '866', '186', '156', '776', '775', '998', '333', '325', '572', '927', '744', '777', '833', '551', '301', '716', '848', '102', '790', '959', '404', '987', '415', '455', '242', '600', '517', '16', '320', '632', '568', '338', '216', '331', '726', '959', '605', '134', '677', '288', '10', '718', '852', '440', '104', '712', '388', '261', '609', '620', '341', '579', '450', '628', '217', '878', '763', '209', '126', '663', '864', '232', '776', '942', '336', '733', '681', '512', '78', '668', '699', '746', '46', '618', '330', '615', '427', '62', '116', '127', '955', '306', '425', '190', '370', '187', '971', '534', '397', '657', '840', '718', '116', '836', '994', '419', '764', '214', '285', '641', '951', '882', '13', '829', '624', '216', '665', '521', '268', '468', '418', '728', '356', '449', '194', '362', '948', '924', '249', '524', '992', '571', '283', '608', '129', '486', '859', '498', '21', '467', '591', '924', '556', '97', '898', '586', '10', '202', '67', '501', '141', '603', '727', '101', '995', '278', '964', '240', '423', '634', '533', '424', '451', '555', '732', '514', '803', '300', '551', '753', '411', '315', '963', '129', '389', '601', '526', '839', '299', '578', '112', '960', '632', '867', '273', '61', '427', '367', '924', '413', '34', '773', '654', '131', '874', '282', '891', '956', '201', '267', '969', '200', '673', '423', '907', '57', '27', '459', '863', '322', '934', '663', '424', '687', '837', '958', '645', '120', '306', '930', '121', '694', '524', '205', '137', '849', '681', '380', '586', '916', '478', '182', '874', '715', '590', '111', '19', '161', '915', '730', '678', '822', '818', '699', '601', '673', '233', '501', '624', '679', '400', '581', '665', '903', '622', '585', '800', '899', '669', '81', '746', '595', '935', '668', '295', '893', '266', '628', '987', '367', '294', '727', '12', '876', '186', '589', '70', '129', '454', '17', '946', '200', '181', '163', '80', '940', '587', '21', '198', '25', '932', '339', '480', '764', '883', '454', '807', '287', '868', '614', '814', '591', '919', '508', '479', '452', '155', '41', '163', '606', '7', '269', '576', '858', '506', '23', '447', '397', '595', '753', '5', '186', '667', '305', '46', '303', '673', '927', '91', '34', '757', '406', '390', '76', '517', '806', '330', '34', '130', '103', '56', '545', '78', '31', '372', '235', '516', '159', '187', '930', '888', '96', '444', '991', '872', '302', '566', '33', '818', '694', '406', '20', '18', '371', '320', '724', '994', '730', '613', '105', '878', '311', '883', '367', '243', '627', '944', '43', '412', '921', '332', '474', '276', '629', '917', '77', '643', '556', '998', '328', '723', '161', '250', '1', '919', '392', '264', '408', '633', '495', '188', '884', '335', '795', '241', '842', '71', '862', '254', '27', '409', '444', '433', '324', '322', '688', '579', '562', '917', '803', '863', '44', '719', '16', '384', '328', '348', '194', '678', '593', '9', '371', '25', '913', '667', '104', '72', '223', '268', '283', '477', '53', '465', '100', '543', '133', '159', '439', '151', '355', '392', '577', '72', '383', '846', '145', '109', '988', '824', '293', '411', '718', '484', '966', '259', '344', '132', '128', '154', '210', '508', '445', '138', '233', '376', '720', '464', '960', '999', '455', '613', '314', '993', '17', '759', '843', '721', '331', '508', '432', '778', '370', '468', '489', '375', '263', '164', '418', '377', '878', '283', '838', '442', '380', '641', '628', '592', '59', '224', '587', '724', '207', '228', '8', '962', '575', '988', '889', '551', '990', '141', '120', '207', '121', '946', '463', '786', '167', '256', '986', '28', '283', '834', '720', '411', '80', '173', '29', '606', '636', '156', '91', '734', '569', '458', '84', '230', '274', '707', '75', '965', '260', '978', '767', '372', '717', '764', '96', '958', '857', '327', '140', '88', '156', '137', '842', '556', '492', '771', '653', '484', '269', '787', '827', '644', '393', '386', '654', '137', '715', '906', '724', '823', '516', '64', '850', '321', '611', '392', '509', '207', '655', '397', '949', '188', '830', '750', '259', '294', '450', '511', '477', '255', '814', '781', '177', '654', '911', '680', '769', '830', '273', '24', '463', '321', '480', '331', '21', '556', '481', '420', '179', '216', '212', '152', '333', '646', '152', '635', '128', '993', '351', '928', '266', '830', '371', '335', '319', '786', '816', '334', '509', '444', '155', '902', '527', '817', '346', '848', '173', '438', '443', '745', '374', '548', '552', '619', '411', '451', '278', '715', '629', '855', '694', '709', '611', '168', '113', '782', '974', '147', '69', '546', '11', '543', '629', '127', '413', '349', '345', '922', '484', '78', '200', '399', '192', '543', '89', '423', '323', '764', '970', '829', '645', '542', '809', '195', '732', '474', '915', '820', '238', '643', '978', '214', '844', '717', '925', '57', '911', '444', '827', '245', '250', '930', '791', '401', '648', '773', '978', '875', '637', '975', '586', '873', '472', '978', '909', '102', '878', '784', '398', '355', '643', '279', '92', '523', '50', '510', '765', '281', '870', '748', '253', '749', '754', '452', '775', '261', '562', '506', '289', '827', '456', '449', '117', '97', '101', '291', '346', '809', '997', '168', '896', '714', '126', '593', '8', '432', '72', '158', '958', '662', '945', '47', '919', '427', '924', '185', '685', '124', '660', '449', '434', '178', '356', '128', '819', '157', '404', '23', '943', '155', '756', '797', '916', '254', '9', '471', '577', '255', '882', '654', '261', '931', '950', '360', '246', '872', '982', '519', '826', '556', '220', '928', '402', '911', '601', '179', '639', '302', '767', '778', '697', '178', '500', '912', '557', '745', '611', '401', '571', '621', '206', '89', '394', '481', '627', '333', '701', '644', '364', '450', '972', '199', '621', '795', '265', '118', '705', '565', '519', '641', '75', '757', '590', '749', '374', '986', '76', '83', '14', '945', '683', '770', '318', '268', '429', '269', '828', '503', '150', '344', '858', '45', '959', '884', '333', '953', '86', '204', '62', '960', '222', '178', '178', '274', '21', '552', '147', '555', '566', '74', '264', '399', '285', '768', '296', '327', '502']

true_classes = ['230', '809', '334', '332', '109', '286', '370', '757', '595', '147', '23', '478', '517', '334', '948', '727', '23', '846', '270', '324', '981', '586', '887', '129', '198', '256', '565', '717', '29', '844', '591', '468', '259', '994', '588', '46', '390', '101', '870', '149', '476', '80', '159', '275', '175', '461', '970', '160', '498', '487', '270', '373', '330', '142', '473', '159', '70', '720', '227', '959', '646', '645', '483', '457', '352', '934', '283', '802', '276', '751', '328', '969', '163', '328', '726', '977', '620', '637', '39', '277', '763', '646', '647', '504', '687', '666', '583', '158', '825', '212', '659', '257', '436', '140', '248', '339', '230', '627', '867', '771', '118', '396', '934', '16', '548', '993', '704', '457', '606', '922', '284', '889', '475', '984', '16', '77', '610', '254', '636', '662', '473', '25', '463', '741', '289', '425', '973', '167', '121', '532', '678', '125', '349', '13', '179', '989', '660', '533', '27', '644', '176', '694', '798', '925', '821', '92', '897', '612', '610', '283', '881', '426', '554', '403', '869', '0', '866', '64', '69', '176', '991', '1', '934', '572', '979', '654', '702', '24', '102', '508', '361', '280', '65', '21', '525', '938', '733', '463', '887', '380', '982', '305', '355', '503', '495', '472', '293', '816', '904', '475', '431', '260', '130', '627', '622', '696', '300', '37', '133', '465', '592', '741', '694', '546', '208', '959', '834', '879', '649', '228', '621', '107', '598', '420', '133', '185', '471', '230', '974', '852', '267', '359', '484', '510', '935', '310', '270', '598', '199', '998', '97', '856', '398', '319', '549', '92', '765', '945', '160', '619', '722', '674', '468', '675', '636', '721', '16', '199', '175', '350', '361', '594', '734', '767', '746', '788', '346', '153', '739', '915', '152', '943', '849', '712', '100', '546', '744', '141', '39', '993', '190', '888', '584', '341', '875', '359', '388', '894', '437', '517', '372', '286', '915', '529', '416', '376', '147', '902', '26', '398', '175', '270', '540', '607', '495', '801', '166', '298', '832', '39', '219', '863', '794', '983', '238', '498', '223', '521', '260', '491', '220', '680', '542', '961', '99', '143', '796', '986', '367', '812', '705', '781', '858', '437', '11', '625', '874', '600', '133', '499', '834', '298', '900', '833', '471', '288', '850', '273', '954', '156', '333', '325', '572', '927', '777', '833', '551', '301', '716', '848', '102', '959', '404', '455', '242', '600', '517', '16', '632', '605', '134', '677', '288', '852', '104', '388', '620', '579', '450', '628', '217', '126', '864', '232', '776', '942', '336', '733', '78', '668', '699', '746', '330', '615', '62', '127', '955', '306', '425', '190', '370', '971', '534', '657', '840', '718', '116', '994', '214', '285', '641', '951', '13', '829', '216', '521', '468', '728', '449', '194', '362', '571', '283', '608', '129', '486', '859', '21', '97', '586', '10', '67', '603', '727', '101', '278', '533', '451', '514', '551', '753', '411', '963', '129', '389', '839', '299', '578', '112', '632', '867', '61', '427', '413', '773', '654', '282', '891', '27', '863', '687', '645', '306', '205', '137', '849', '586', '916', '478', '874', '111', '19', '915', '822', '699', '601', '673', '233', '624', '679', '581', '665', '622', '669', '81', '746', '935', '668', '295', '893', '628', '727', '876', '589', '70', '129', '454', '17', '946', '181', '163', '80', '21', '198', '25', '932', '339', '480', '883', '614', '591', '919', '508', '479', '452', '41', '606', '506', '23', '595', '753', '667', '305', '46', '927', '91', '34', '757', '76', '517', '806', '78', '31', '372', '888', '96', '566', '33', '694', '406', '20', '18', '371', '320', '730', '613', '311', '367', '243', '412', '921', '332', '629', '917', '77', '556', '998', '328', '723', '161', '250', '1', '919', '392', '264', '495']