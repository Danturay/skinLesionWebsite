import Link from "next/link";

const NavLink = ({
  href,
  title,
  onClick,
}: {
  href: string;
  title: string;
  onClick?: () => void;
}) => {
  return (
    <Link
      href={href}
      onClick={onClick} // ✅ Now supported
      className="block py-2 pl-3 pr-4 text-[#ADB7BE] sm:text-xl rounded md:p-0 hover:text-white"
    >
      {title}
    </Link>
  );
};

export default NavLink;
